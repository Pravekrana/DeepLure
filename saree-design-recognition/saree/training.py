import time
from pathlib import Path
import numpy as np
import torch
from .common import read_json,write_json,digest,seed_all,safe_path,object_hash
from .data import assert_no_leakage,verify_files
from .model import Embedder,save_model,load_model
from .transforms import Preprocess,load_image
from .losses import supervised_contrastive
from .sampling import PKSampler
from .retrieval import protocol_embeddings
from .metrics import retrieval_metrics


def train(root,split_file,out,config,device='cpu'):
    cfg=read_json(config) if isinstance(config,(str,Path)) else config
    seed_all(cfg['seed']);spec=read_json(split_file);assert_no_leakage(spec)
    rows=[r for r in spec['rows'] if r['split']=='train' and r['role']=='train']
    if len(rows)<2: raise ValueError('At least two eligible training images are required')
    verify_files([r for r in spec['rows'] if r['split'] in ('train','val') and r['role']!='excluded'],root)
    mode=spec['mode'];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists(): raise FileExistsError('Training output already contains checkpoint; choose new output')
    pretrained=cfg.get('pretrained',True)
    if not pretrained and not cfg.get('synthetic_smoke',False): raise ValueError('Random initialization allowed only in explicit synthetic smoke mode')
    model=Embedder('trained',pretrained).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    transform=Preprocess(train=True,color=cfg['color_augmentation'])
    # Decode once for bounded datasets; image memory counted separately from model efficiency.
    images=[load_image(safe_path(root,r['relative_path'])) for r in rows]
    sampler=PKSampler(rows,cfg['p'],cfg['k'],cfg['seed']) if mode=='supervised' else None
    ids=sorted(set(r['design_id'] for r in rows if r['design_id'])); labelmap={d:i for i,d in enumerate(ids)}
    best=-1.;logs=[];start=time.monotonic();stopped=False
    use_amp=bool(cfg.get('amp',False) and torch.device(device).type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    for epoch in range(cfg['epochs']):
        freeze=epoch<cfg['warmup_epochs']
        for p in model.features.parameters(): p.requires_grad=not freeze
        model.train()
        # Fixed BN running statistics on small batches; affine weights fine-tune after warm-up.
        model.features.eval()
        if sampler: batches=list(sampler.batches(epoch))
        else:
            order=np.random.default_rng(cfg['seed']+epoch).permutation(len(rows)).tolist()
            batches=[order[i:i+cfg['batch_size']] for i in range(0,len(order),cfg['batch_size'])]
        epoch_losses=[];grad_norms=[]
        for inds in batches:
            if len(inds)<2: continue
            x=torch.stack([transform(images[i]) for i in inds for _ in range(2)]).to(device)
            labels=torch.tensor([labelmap[rows[i]['design_id']] if sampler else i for i in inds for _ in range(2)],device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type,enabled=use_amp):
                z=model(x);loss=supervised_contrastive(z,labels,cfg['temperature'])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
            if not torch.isfinite(norm) or norm<=0: raise RuntimeError('Nonfinite or zero gradients')
            scaler.step(opt);scaler.update()
            epoch_losses.append(float(loss.detach()));grad_norms.append(float(norm))
            if time.monotonic()-start>cfg['max_seconds']: stopped=True;break
        if not epoch_losses: raise RuntimeError('No valid training batches')
        model.eval()
        q,g,ql,gl=protocol_embeddings(model,spec,root,'val',device)
        score=retrieval_metrics(q@g.T,ql,gl)['mAP']
        spread=float(np.std(g,axis=0).mean())
        if not np.isfinite(g).all() or spread<1e-6: raise RuntimeError('Collapsed/nonfinite validation embeddings')
        log={'epoch':epoch+1,'loss':float(np.mean(epoch_losses)),'gradient_norm_mean':float(np.mean(grad_norms)),
             'validation_mAP':score,'validation_embedding_std_mean':spread,'elapsed_seconds':time.monotonic()-start,
             'backbone_frozen':freeze,'batches':len(epoch_losses)}
        logs.append(log);print(log,flush=True)
        meta={'config':cfg,'split_sha256':digest(split_file),'manifest_sha256':spec['manifest_sha256'],
              'config_sha256':object_hash(cfg),'epoch':epoch+1,'selection':'validation mAP; earliest epoch on ties',
              'mode':mode,'validation_protocol':spec['protocols']['val']['name'],
              'initialization':'MobileNet_V3_Small_Weights.IMAGENET1K_V1' if pretrained else 'random_synthetic_smoke_only',
              'train_images':len(rows),'precision':'mixed_float16' if use_amp else 'float32'}
        save_model(out/'last.pt',model,meta,opt)
        if score>best:
            best=score;save_model(out/'best.pt',model,meta,opt)
        write_json(out/'training_log.json',logs);write_json(out/'config.json',cfg)
        if stopped: break
    loaded,_=load_model(out/'best.pt',device)
    # Verify byte-identical embedding definition survives save/load for selected weights.
    a,_=load_model(out/'best.pt',device)
    x=Preprocess()(images[0]).unsqueeze(0).to(device)
    with torch.no_grad(): delta=float((loaded(x)-a(x)).abs().max())
    write_json(out/'training_summary.json',{'epochs_executed':len(logs),'best_validation_mAP':best,
               'seconds':time.monotonic()-start,'time_budget_reached':stopped,'reload_max_abs_difference':delta,
               'test_used':False,'real_design_claims_supported':mode=='supervised'})
    return str(out/'best.pt')


def baseline(kind,split_file,out):
    if kind not in ('rgb','gray'): raise ValueError('Frozen baseline must be rgb or gray')
    seed_all(17);m=Embedder(kind,pretrained=True).eval();spec=read_json(split_file)
    save_model(out,m,{'mode':spec['mode'],'split_sha256':digest(split_file),'manifest_sha256':spec['manifest_sha256'],
                     'initialization':'MobileNet_V3_Small_Weights.IMAGENET1K_V1','frozen':True,
                     'features':'actual 576-D average pooled backbone; no projection head'})

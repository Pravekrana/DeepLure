from pathlib import Path
import numpy as np
from .common import read_json,write_json,digest
from .data import assert_no_leakage
from .model import load_model
from .retrieval import protocol_embeddings
from .metrics import retrieval_metrics,verification_metrics,select_threshold


def pair_scores(spec,split,scores):
    p=spec['protocols'][split]; qi={v:i for i,v in enumerate(p['query'])};gi={v:i for i,v in enumerate(p['gallery'])}
    pairs=p['pairs']
    return [r['label'] for r in pairs],[float(scores[qi[r['a']],gi[r['b']]]) for r in pairs]


def calibrate(checkpoint,split_file,root,out,device='cpu'):
    spec=read_json(split_file);assert_no_leakage(spec);model,meta=load_model(checkpoint,device)
    if meta['split_sha256']!=digest(split_file): raise ValueError('Checkpoint/split mismatch')
    q,g,ql,gl=protocol_embeddings(model,spec,root,'val',device)
    scores=q@g.T; labels,values=pair_scores(spec,'val',scores)
    cal=select_threshold(labels,values,'val')
    cal.update(checkpoint_sha256=digest(checkpoint),split_sha256=digest(split_file),
               pair_list_sha256=__import__('saree.common',fromlist=['object_hash']).object_hash(spec['protocols']['val']['pairs']),
               protocol=spec['protocols']['val']['name'],retrieval=retrieval_metrics(scores,ql,gl),
               verification=verification_metrics(labels,values,cal['threshold']))
    if spec['mode']=='instance':
        cal['secondary_stresses']={}
        for stress in ('grayscale','inversion'):
            sq,sg,sql,sgl=protocol_embeddings(model,spec,root,'val',device,stress)
            sl,sv=pair_scores(spec,'val',sq@sg.T)
            cal['secondary_stresses'][stress]={'retrieval':retrieval_metrics(sq@sg.T,sql,sgl),
                'verification':verification_metrics(sl,sv,cal['threshold'])}
    write_json(out,cal);return cal


def lock_models(split_file,checkpoints,calibrations,out):
    if Path(out).exists(): raise FileExistsError('Experiment lock already exists')
    entries=[]
    for ck,cal in zip(checkpoints,calibrations):
        c=read_json(cal)
        if c['checkpoint_sha256']!=digest(ck) or c['split_sha256']!=digest(split_file): raise ValueError('Calibration provenance mismatch')
        entries.append({'checkpoint':str(ck),'sha256':digest(ck),'calibration':str(cal),'calibration_sha256':digest(cal)})
    if not entries or len(checkpoints)!=len(calibrations): raise ValueError('Provide matching checkpoint/calibration lists')
    write_json(out,{'split_sha256':digest(split_file),'models':entries,'test_policy':'One final test invocation after validation decisions locked',
                    'ranking':'image-level cosine; stable manifest-order tie break','synthetic_transforms':['RGB channel permutation','grayscale','RGB complement 255-x']})


def evaluate(lock,split_file,root,out,device='cpu'):
    lock=read_json(lock);spec=read_json(split_file);assert_no_leakage(spec)
    if lock['split_sha256']!=digest(split_file): raise ValueError('Frozen split hash mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() or (out/'TEST_STARTED.json').exists():
        raise FileExistsError('Test already started/executed. Document any implementation correction before a new output directory.')
    # Validate all assets before marking final test as accessed.
    for e in lock['models']:
        if digest(e['checkpoint'])!=e['sha256'] or digest(e['calibration'])!=e['calibration_sha256']: raise ValueError('Locked asset changed')
    write_json(out/'TEST_STARTED.json',{'lock':lock,'status':'started'})
    results={}
    for e in lock['models']:
        model,meta=load_model(e['checkpoint'],device);cal=read_json(e['calibration'])
        name=Path(e['checkpoint']).parent.name+'_'+model.kind
        result={'checkpoint_sha256':e['sha256'],'kind':model.kind,'protocol':spec['protocols']['test']['name'],
                'natural_cross_colorway':{'available':False,'reason':'No established colorway labels in executed instance protocol'},
                'real_design_verification':{'available':spec['mode']=='supervised','reason':None if spec['mode']=='supervised' else 'No trusted distinct-image design identities'},
                'hard_different_design_similar_palette':{'available':False,'reason':'Requires verified different-design identities and palette subset annotation'},
                'excluded_images':sum(r['split']=='test' and r['role']=='excluded' for r in spec['rows'])}
        for stress in (['recolor','grayscale','inversion'] if spec['mode']=='instance' else ['natural']):
            q,g,ql,gl=protocol_embeddings(model,spec,root,'test',device,stress)
            scores=q@g.T; labels,values=pair_scores(spec,'test',scores)
            result[stress]={'retrieval':retrieval_metrics(scores,ql,gl),
                            'verification':verification_metrics(labels,values,cal['threshold']),
                            'threshold_note':'Same validation-recolor threshold used unchanged for secondary grayscale/inversion stresses',
                            'embedding_std_mean':float(g.std(0).mean())}
            order=np.argsort(-scores,axis=1,kind='stable')
            errors=[{'query_id':spec['protocols']['test']['query'][i],
                     'top_image_id':spec['protocols']['test']['gallery'][int(order[i,0])],
                     'top_score':float(scores[i,order[i,0]]),'expected_identity':ql[i]}
                    for i in range(len(q)) if gl[order[i,0]]!=ql[i]]
            write_json(out/f'{name}_{stress}_errors.private.json',errors)
            np.savez(out/f'{name}_{stress}_scores.private.npz',scores=scores)
            if spec['mode']=='supervised':
                lookup={r['image_id']:r for r in spec['rows']};p=spec['protocols']['test']
                eligible=[]
                for i,qid in enumerate(p['query']):
                    qr=lookup[qid]
                    if qr['colorway_id'] and any(lookup[gid]['design_id']==qr['design_id'] and lookup[gid]['colorway_id'] and lookup[gid]['colorway_id']!=qr['colorway_id'] for gid in p['gallery']): eligible.append(i)
                if eligible:
                    # Only different-colorway positives remain eligible in each query ranking.
                    subset=[]
                    for i in eligible:
                        qr=lookup[p['query'][i]]
                        keep=[j for j,gid in enumerate(p['gallery']) if lookup[gid]['design_id']!=qr['design_id'] or (lookup[gid]['colorway_id'] and lookup[gid]['colorway_id']!=qr['colorway_id'])]
                        subset.append(retrieval_metrics(scores[i:i+1,keep],[ql[i]],[gl[j] for j in keep]))
                    result['natural_cross_colorway']={'available':True,'queries':len(eligible),'recall_at_1':float(np.mean([x['recall_at_1'] for x in subset])),'mAP':float(np.mean([x['mAP'] for x in subset]))}
        results[name]=result
    write_json(out/'metrics.json',{'split_sha256':digest(split_file),'mode':spec['mode'],'models':results,
       'limitations':['Synthetic recolored copies are not distinct captures or manufactured colorways.',
                      'Unknown identities can cause false negatives in instance training/evaluation.',
                      'No low-FPR reliability claims; pairs are dependent.'] if spec['mode']=='instance' else []})
    plot_results(results,out)
    return results


def plot_results(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    lines=['model,protocol,recall_at_1,recall_at_5,mAP,roc_auc,precision,recall,f1,balanced_accuracy']
    for name,r in results.items():
        for stress in ('recolor','grayscale','inversion','natural'):
            if stress not in r: continue
            a=r[stress]['retrieval'];b=r[stress]['verification']
            lines.append(','.join(map(str,[name,stress,a['recall_at_1'],a['recall_at_5'],a['mAP'],b.get('roc_auc'),b.get('precision'),b.get('recall'),b.get('f1'),b.get('balanced_accuracy')])))
            if b['available']: ax.plot(b['roc']['fpr'],b['roc']['tpr'],label=f'{name}/{stress}')
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Verification ROC — protocol labels, not probabilities');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(Path(out)/'roc.png');plt.close(fig)
    (Path(out)/'results.csv').write_text('\n'.join(lines)+'\n')

from pathlib import Path
import numpy as np
import torch
from .common import read_json, write_json, safe_path, digest
from .model import load_model, identity
from .transforms import Preprocess, load_image, recolor


@torch.inference_mode()
def encode(model, images, device='cpu', batch_size=16, transform=None):
    transform=transform or Preprocess(grayscale=model.kind=='gray')
    result=[]; batch=[]
    for im in images:
        batch.append(transform(im))
        if len(batch)==batch_size:
            result.append(model(torch.stack(batch).to(device)).cpu().numpy());batch=[]
    if batch: result.append(model(torch.stack(batch).to(device)).cpu().numpy())
    return np.concatenate(result).astype(np.float32) if result else np.empty((0,model.dim),dtype=np.float32)


def protocol_embeddings(model, spec, root, split, device='cpu', stress='recolor'):
    protocol=spec['protocols'][split]; lookup={r['image_id']:r for r in spec['rows']}
    gallery=protocol['gallery'];query=protocol['query']
    if not gallery or not query: raise ValueError(f'No eligible query/gallery in {split}')
    from .data import verify_files
    verify_files([lookup[i] for i in set(gallery+query)],root)
    g=encode(model,(load_image(safe_path(root,lookup[i]['relative_path'])) for i in gallery),device)
    def queries():
        from PIL import ImageOps
        for j,i in enumerate(query):
            im=load_image(safe_path(root,lookup[i]['relative_path']))
            if spec['mode']=='instance':
                if stress=='recolor': im=recolor(im,protocol['recolor_seed']+j)
                elif stress=='grayscale': im=ImageOps.grayscale(im).convert('RGB')
                elif stress=='inversion': im=ImageOps.invert(im)
                else: raise ValueError('Unknown synthetic stress transform')
            yield im
    q=encode(model,queries(),device)
    labels=lambda ids:[lookup[i]['design_id'] if spec['mode']=='supervised' else i for i in ids]
    return q,g,labels(query),labels(gallery)


def build_gallery(checkpoint, split_file, root, out, split='test', device='cpu'):
    model,meta=load_model(checkpoint,device);spec=read_json(split_file)
    ids=spec['protocols'][split]['gallery'];lookup={r['image_id']:r for r in spec['rows']}
    if not ids: raise ValueError('Gallery is empty')
    from .data import verify_files
    verify_files([lookup[i] for i in ids],root)
    z=encode(model,(load_image(safe_path(root,lookup[i]['relative_path'])) for i in ids),device)
    Path(out).mkdir(parents=True,exist_ok=True)
    np.save(Path(out)/'embeddings.npy',z)
    cache={**identity(checkpoint),'split_sha256':digest(split_file),'manifest_sha256':spec['manifest_sha256'],
           'embedding_sha256':digest(Path(out)/'embeddings.npy'),'split':split,'ids':ids,
           'design_ids':[lookup[i]['design_id'] for i in ids],'dim':model.dim,'dtype':'float32',
           'protocol':spec['protocols'][split]['name'],'checkpoint_metadata':meta}
    write_json(Path(out)/'cache.json',cache)
    return cache


def load_gallery(checkpoint, folder, split_file):
    cache=read_json(Path(folder)/'cache.json')
    for k,v in identity(checkpoint).items():
        if cache[k]!=v: raise ValueError(f'Incompatible gallery cache: {k}')
    if cache['split_sha256']!=digest(split_file): raise ValueError('Incompatible gallery split manifest')
    path=Path(folder)/'embeddings.npy'
    if cache['embedding_sha256']!=digest(path): raise ValueError('Gallery embedding hash mismatch')
    z=np.load(path,allow_pickle=False)
    if z.shape!=(len(cache['ids']),cache['dim']) or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5):
        raise ValueError('Invalid gallery embeddings')
    return z,cache


def retrieve(checkpoint,folder,split_file,image,k=5,device='cpu'):
    model,meta=load_model(checkpoint,device);g,cache=load_gallery(checkpoint,folder,split_file)
    im=load_image(image) if isinstance(image,(str,Path)) else image
    q=encode(model,[im],device); scores=(q@g.T)[0]
    order=np.argsort(-scores,kind='stable')[:max(1,k)]
    return [{'image_id':cache['ids'][i],'design_id':cache['design_ids'][i],'cosine_similarity':float(scores[i])} for i in order]


def verify(checkpoint,calibration,a,b,device='cpu'):
    model,meta=load_model(checkpoint,device);cal=read_json(calibration)
    if cal['checkpoint_sha256']!=digest(checkpoint) or cal['selected_on']!='val':
        raise ValueError('Threshold provenance incompatible with checkpoint/validation')
    images=[load_image(x) if isinstance(x,(str,Path)) else x for x in (a,b)]
    z=encode(model,images,device);score=float(z[0]@z[1])
    return {'cosine_similarity':score,'threshold':cal['threshold'],'same_design_decision':score>=cal['threshold'],
            'calibration_protocol':cal['protocol'],'claim_supported':cal['protocol']=='natural_distinct_capture',
            'warning':None if cal['protocol']=='natural_distinct_capture' else 'Synthetic instance calibration only; genuine same-design decision unvalidated'}

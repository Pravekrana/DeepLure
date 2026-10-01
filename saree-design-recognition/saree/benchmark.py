import platform
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import write_json,digest,seed_all
from .model import load_model
from .transforms import Preprocess


def benchmark(checkpoint,out,device='cpu',warmup=10,repeats=50,gallery_size=1000):
    seed_all(17);model,meta=load_model(checkpoint,device)
    def sync():
        if torch.device(device).type=='cuda': torch.cuda.synchronize()
        elif torch.device(device).type=='mps': torch.mps.synchronize()
    x=torch.zeros(1,3,224,224,device=device)
    times=[]
    with torch.inference_mode():
        for _ in range(warmup): model(x)
        sync()
        for _ in range(repeats):
            sync();t=time.perf_counter();model(x);sync();times.append((time.perf_counter()-t)*1000)
    im=Image.new('RGB',(640,480),'gray');tx=Preprocess(grayscale=model.kind=='gray');prep=[]
    for _ in range(warmup): tx(im)
    for _ in range(repeats):
        t=time.perf_counter();tx(im);prep.append((time.perf_counter()-t)*1000)
    rng=np.random.default_rng(17);g=rng.normal(size=(gallery_size,model.dim)).astype(np.float32);g/=np.linalg.norm(g,axis=1,keepdims=True)
    q=g[:1];search=[]
    for _ in range(warmup): np.argsort(-(q@g.T)[0])[:5]
    for _ in range(repeats):
        t=time.perf_counter();np.argsort(-(q@g.T)[0])[:5];search.append((time.perf_counter()-t)*1000)
    stats=lambda a:{'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),'mean_ms':float(np.mean(a))}
    result={'checkpoint_sha256':digest(checkpoint),'parameters':sum(p.numel() for p in model.parameters()),
            'embedding_dimension':model.dim,'embedding_dtype':'float32','embedding_bytes':model.dim*4,
            'checkpoint_bytes':Path(checkpoint).stat().st_size,'checkpoint_contains_optimizer':meta.get('frozen') is not True,
            'device':str(device),'hardware':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.machine()) if Path('/proc/cpuinfo').exists() else platform.machine(),'platform':platform.platform(),
            'torch':str(torch.__version__),'input_shape':[1,3,224,224],'batch_size':1,'precision':'float32',
            'threads':torch.get_num_threads(),'warmup':warmup,'repeats':repeats,'accelerator_synchronized':torch.device(device).type!='cpu',
            'encoder':stats(times),'preprocessing_excludes_disk_decode':stats(prep),'preprocess_source_shape':[480,640,3],'search':stats(search),
            'search_gallery_size':gallery_size,'search_gallery':'generated normalized vectors; matrix multiply plus exact sort on CPU',
            'flops':None,'flops_reason':'Not measured; no counting tool used'}
    write_json(out,result);return result

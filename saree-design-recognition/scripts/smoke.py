"""Executable train/calibrate/lock/test/gallery/verify integration on generated fixtures."""
import argparse
from pathlib import Path
from saree.common import write_json,read_json,seed_all
from saree.fixtures import generate
from saree.data import audit,make_splits
from saree.training import train
from saree.evaluation import calibrate,lock_models,evaluate
from saree.retrieval import build_gallery,retrieve,verify
from saree.benchmark import benchmark

p=argparse.ArgumentParser();p.add_argument('--out',default='private/smoke');args=p.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True);root=out/'data'
generate(root);audit(root,out/'manifest.json');make_splits(out/'manifest.json',out/'split.json')
cfg=read_json('configs/cpu.json');cfg.update(epochs=2,batch_size=4,max_seconds=120,pretrained=True,synthetic_smoke=True)
ck=train(root,out/'split.json',out/'trained',cfg)
calibrate(ck,out/'split.json',root,out/'calibration.json')
lock_models(out/'split.json',[ck],[out/'calibration.json'],out/'lock.json')
evaluate(out/'lock.json',out/'split.json',root,out/'evaluation')
build_gallery(ck,out/'split.json',root,out/'gallery')
result=retrieve(ck,out/'gallery',out/'split.json',root/'fixture_000.png')
v=verify(ck,out/'calibration.json',root/'fixture_000.png',root/'fixture_001.png')
benchmark(ck,out/'efficiency.json',repeats=10)
write_json(out/'smoke_summary.json',{'executed':True,'real_dataset':False,'retrieval_returned':len(result),'verification':v})
print('Synthetic software integration completed; not real design-recognition evidence.')

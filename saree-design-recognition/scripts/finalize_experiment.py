"""Calibrate on validation, freeze assets, then run the final test exactly once."""
import argparse
from pathlib import Path
from saree.common import device_for,seed_all
from saree.evaluation import calibrate,lock_models,evaluate
from saree.retrieval import build_gallery
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--split',required=True);p.add_argument('--device',default='auto');a=p.parse_args()
seed_all(17);root=Path(a.run_root);device=device_for(a.device)
if (root/'experiment_lock.json').exists():raise SystemExit('Experiment already locked; preserve it and use a new run for changed validation decisions.')
checkpoints=[root/'trained/best.pt',root/'rgb/baseline.pt',root/'gray/baseline.pt',root/'no_color/best.pt']
for ck in checkpoints:
    if not ck.is_file():raise FileNotFoundError(f'Missing model asset: {ck}')
calibrations=[]
for ck in checkpoints:
    cal=ck.parent/'calibration.json';result=calibrate(ck,a.split,a.data_root,cal,device);calibrations.append(cal)
    print(ck.parent.name,'validation mAP',result['retrieval']['mAP'],'threshold',result['threshold'],flush=True)
lock_models(a.split,checkpoints,calibrations,root/'experiment_lock.json')
evaluate(root/'experiment_lock.json',a.split,a.data_root,root/'evaluation',device)
build_gallery(checkpoints[0],a.split,a.data_root,root/'gallery',device=device)
print('Final test complete. Natural cross-colorway claims depend on actual annotation coverage, not synthetic scores.')

"""Dry-run by default; only deletes project-created private directory after exercise."""
import argparse
import shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--exercise-concluded',action='store_true');p.add_argument('--delete-project-copies',action='store_true');args=p.parse_args()
project=Path(__file__).resolve().parents[1];target=project/'private';marker=target/'.project-created'
if target.is_symlink() or not marker.is_file():raise SystemExit('Refusing cleanup: project ownership marker missing or private directory is symlink')
temporary_files=[Path('/tmp')/n for n in ['deeplure_sample','deeplure_retry.jpg','deeplure_drive.html','deeplure_sub1.html','deeplure_sub2.html','deeplure_kaggle.json']]
print('Additional exact temporary acquisition files:',[str(p) for p in temporary_files if p.is_file()])
print('Scoped target:',target,'(all downloaded copies, crops, manifests, caches and local checkpoints)')
if args.delete_project_copies:
    if not args.exercise_concluded:raise SystemExit('Exercise must be concluded before deletion')
    for p in temporary_files:
        if p.is_file() and not p.is_symlink(): p.unlink()
    shutil.rmtree(target);print('Deleted project copies only. Original ps_v3.pdf and external/user datasets were untouched.')
else:print('Dry run only; no deletion. Add both flags after the exercise concludes.')

import argparse
import json
from pathlib import Path
from .common import device_for,seed_all


def main(argv=None):
    p=argparse.ArgumentParser(description='Private local saree embedding pipeline; no automatic synthetic fallback')
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--data-root',required=True);a.add_argument('--out',required=True);a.add_argument('--annotations')
    a=sub.add_parser('split');a.add_argument('--manifest',required=True);a.add_argument('--out',required=True);a.add_argument('--mode',choices=['instance','supervised'],required=True);a.add_argument('--seed',type=int,default=17);a.add_argument('--max-per-split',type=int)
    a=sub.add_parser('fixtures');a.add_argument('--out',required=True)
    a=sub.add_parser('train');a.add_argument('--data-root',required=True);a.add_argument('--split',required=True);a.add_argument('--out',required=True);a.add_argument('--config',required=True)
    a=sub.add_parser('baseline');a.add_argument('--kind',choices=['rgb','gray'],required=True);a.add_argument('--split',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('calibrate');a.add_argument('--checkpoint',required=True);a.add_argument('--split',required=True);a.add_argument('--data-root',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('lock');a.add_argument('--split',required=True);a.add_argument('--checkpoints',nargs='+',required=True);a.add_argument('--calibrations',nargs='+',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('evaluate');a.add_argument('--lock',required=True);a.add_argument('--split',required=True);a.add_argument('--data-root',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('gallery');a.add_argument('--checkpoint',required=True);a.add_argument('--split',required=True);a.add_argument('--data-root',required=True);a.add_argument('--out',required=True);a.add_argument('--partition',choices=['val','test'],default='test')
    a=sub.add_parser('retrieve');a.add_argument('--checkpoint',required=True);a.add_argument('--split',required=True);a.add_argument('--gallery',required=True);a.add_argument('--image',required=True);a.add_argument('--k',type=int,default=5)
    a=sub.add_parser('verify');a.add_argument('--checkpoint',required=True);a.add_argument('--calibration',required=True);a.add_argument('--a',required=True);a.add_argument('--b',required=True)
    a=sub.add_parser('benchmark');a.add_argument('--checkpoint',required=True);a.add_argument('--out',required=True);a.add_argument('--repeats',type=int,default=50)
    args=p.parse_args(argv);seed_all(17)
    try:
        d=device_for(args.device);c=args.command;result=None
        if c=='audit':
            from .data import audit
            result=audit(args.data_root,args.out,args.annotations)
        elif c=='split':
            from .data import make_splits
            result=make_splits(args.manifest,args.out,args.mode,args.seed,args.max_per_split)
        elif c=='fixtures':
            from .fixtures import generate
            generate(args.out);result={'software_fixtures':args.out}
        elif c=='train':
            from .training import train
            result=train(args.data_root,args.split,args.out,args.config,d)
        elif c=='baseline':
            from .training import baseline
            baseline(args.kind,args.split,args.out);result={'checkpoint':args.out}
        elif c=='calibrate':
            from .evaluation import calibrate
            result=calibrate(args.checkpoint,args.split,args.data_root,args.out,d)
        elif c=='lock':
            from .evaluation import lock_models
            lock_models(args.split,args.checkpoints,args.calibrations,args.out);result={'lock':args.out}
        elif c=='evaluate':
            from .evaluation import evaluate
            result=evaluate(args.lock,args.split,args.data_root,args.out,d)
        elif c=='gallery':
            from .retrieval import build_gallery
            result=build_gallery(args.checkpoint,args.split,args.data_root,args.out,args.partition,d)
        elif c=='retrieve':
            from .retrieval import retrieve
            result=retrieve(args.checkpoint,args.gallery,args.split,args.image,args.k,d)
        elif c=='verify':
            from .retrieval import verify
            result=verify(args.checkpoint,args.calibration,args.a,args.b,d)
        elif c=='benchmark':
            from .benchmark import benchmark
            result=benchmark(args.checkpoint,args.out,d,repeats=args.repeats)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,FileNotFoundError,FileExistsError) as exc:
        p.exit(2,f'Error: {exc}\n')


if __name__=='__main__': main()

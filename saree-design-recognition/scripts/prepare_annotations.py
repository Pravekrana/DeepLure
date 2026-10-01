"""Apply disclosed public-file exclusions and conservative vendor exclusion."""
import argparse
from saree.common import write_json
from saree.data import policy_annotations
p=argparse.ArgumentParser();p.add_argument('--data-root',default='private/data');p.add_argument('--out',default='private/annotations_v2.json');p.add_argument('--policy',default='configs/audit_policy.json');a=p.parse_args()
ann=policy_annotations(a.data_root,a.policy);write_json(a.out,ann);print('Explicit exclusion entries:',len(ann))

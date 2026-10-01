"""Authorized acquisition only. Vendor link must be supplied privately by the user."""
import argparse
import json
import zipfile
from pathlib import Path
import requests

p=argparse.ArgumentParser();p.add_argument('--data-root',default='private/data');p.add_argument('--source',choices=['kaggle','vendor'],required=True);p.add_argument('--vendor-url');args=p.parse_args()
root=Path(args.data_root);root.mkdir(parents=True,exist_ok=True)
if args.source=='kaggle':
    target=root/'kaggle.zip'
    url='https://www.kaggle.com/api/v1/datasets/download/div456/indian-saree-patterns'
    with requests.get(url,stream=True,timeout=120) as r:
        r.raise_for_status()
        with target.open('wb') as f:
            for chunk in r.iter_content(1024*1024):f.write(chunk)
    dest=(root/'kaggle').resolve()
    with zipfile.ZipFile(target) as z:
        for name in z.namelist():
            if not (dest/name).resolve().is_relative_to(dest):raise ValueError('Unsafe archive member')
        z.extractall(dest)
    print('Downloaded Kaggle data; inspect bundled README and license before use.')
else:
    if not args.vendor_url:p.error('--vendor-url must be supplied via the authorized private assignment link')
    import gdown
    files=gdown.download_folder(url=args.vendor_url,output=str(root/'vendor'),use_cookies=False,quiet=True)
    print('Vendor files downloaded:',len(files or []),'; keep private and delete project copies after exercise.')

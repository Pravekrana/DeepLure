"""Explicit reviewed allowlist; no recursive inclusion, data, weights or private outputs."""
import hashlib,json,re,zipfile
from pathlib import Path
import nbformat
root=Path(__file__).resolve().parents[1]
allowed=[s.strip() for s in (root/'SHAREABLE_FILES.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
if len(allowed)!=len(set(allowed)):raise ValueError('Duplicate allowlist entries')
for name in allowed:
    p=(root/name).resolve()
    if not p.is_relative_to(root) or not p.is_file():raise ValueError(f'Invalid allowlisted path: {name}')
    if any(x in Path(name).parts for x in ['private','.venv','.git','.codex','.agents','__pycache__']):raise ValueError(f'Restricted path: {name}')
    if p.suffix in ['.pt','.pth','.npy','.npz','.jpg','.jpeg','.webp']:raise ValueError(f'Restricted asset: {name}')
    if p.suffix=='.ipynb':
        nb=nbformat.read(p,as_version=4)
        for c in nb.cells:
            for o in c.get('outputs',[]):
                data=o.get('data',{})
                if any(k.startswith('image/') or k=='text/html' for k in data):raise ValueError('Image/HTML notebook output cannot be shared')
                if o.get('output_type')=='error':raise ValueError('Notebook has error outputs')
    if p.suffix in ('.md','.txt','.json','.py','.toml','.ipynb','.sh','.csv'):
        text=p.read_text()
        patterns=[r'drive\.google\.com/(?:open|file|drive/folders)',r'AIza[0-9A-Za-z_-]{30,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'(?i)authorization:\s*bearer\s+[A-Za-z0-9]',r'h_img_\d+',r'img_\d+\.jpg']
        if any(re.search(pattern,text) for pattern in patterns):raise ValueError(f'Potential private/access content in {name}')
if len((root/'APPROACH_500_CHARS.txt').read_text())>500:raise ValueError('Approach exceeds 500 characters')
out=root/'dist/DeepLure_Submission.zip';out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in allowed:z.write(root/name,'saree-design-recognition/'+name)
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None:raise ValueError('ZIP CRC validation failed')
    assert set(z.namelist())=={'saree-design-recognition/'+s for s in allowed}
record={'archive':out.name,'files':len(allowed),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'checks':['explicit file allowlist','no private/data/weights/caches','no image-bearing/error notebook outputs','private access pattern scan','ZIP CRC and exact member list','approach <=500 characters'],
        'members':allowed}
(root/'dist/package_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='members'},indent=2))

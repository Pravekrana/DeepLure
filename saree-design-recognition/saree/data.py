"""Private audit manifest and conservative leakage groups. No inferred design labels."""
from collections import Counter, defaultdict
from pathlib import Path
import random
import re
import numpy as np
from PIL import Image
from .common import digest, object_hash, read_json, write_json
from .transforms import load_image


class UnionFind:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, a):
        while a != self.parent[a]:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a
    def join(self, a, b): self.parent[self.find(b)] = self.find(a)


def dhash(im):
    a = np.asarray(im.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
    return int(''.join('1' if v else '0' for v in (a[:, 1:] > a[:, :-1]).flat), 2)


def audit(root, out, annotations=None):
    root = Path(root)
    if not root.is_dir(): raise FileNotFoundError(f'Missing DATA_ROOT: {root}')
    ann = read_json(annotations) if annotations else {}
    rows, corrupt, thumbnails = [], [], []
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'): continue
        rel = p.relative_to(root).as_posix()
        try:
            with Image.open(p) as raw:
                raw.verify()
            with Image.open(p) as raw:
                orientation = raw.getexif().get(274)
            im = load_image(p)
            h = digest(p)
            source = rel.split('/')[0]
            # Roboflow explicitly documents derivatives; names group origins, never designs.
            original = source + '/' + re.sub(r'\.rf\..*$', '', p.name) if '.rf.' in p.name else rel
            a = ann.get(rel, {})
            if a.get('design_id') and (not a.get('annotation_provenance') or a.get('annotation_confidence') != 'high'):
                raise ValueError(f'Identity annotation needs high confidence and provenance: {rel}')
            rows.append(dict(image_id=object_hash(rel)[:16], relative_path=rel, source=source,
                design_id=a.get('design_id'), colorway_id=a.get('colorway_id'),
                original_group=object_hash(original)[:16], capture_group=a.get('capture_group'),
                duplicate_group=None, annotation_provenance=a.get('annotation_provenance', 'none; folder/category is not design'),
                annotation_confidence=a.get('annotation_confidence', 'unknown'), split=None, role=None,
                sha256=h, pixel_sha256=__import__('hashlib').sha256(im.tobytes()).hexdigest(),
                dhash=f'{dhash(im):016x}', width=im.width, height=im.height, exif_orientation=orientation,
                category=Path(rel).parent.name, excluded_reason=a.get('excluded_reason')))
            thumbnails.append(np.asarray(im.resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32)/255)
        except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
            corrupt.append({'relative_path': rel, 'reason': str(exc)})
    if not rows: raise ValueError('No readable images in DATA_ROOT; synthetic fallback is never automatic')
    uf = UnionFind(len(rows)); maps = {k: {} for k in ['sha256', 'pixel_sha256', 'original_group', 'capture_group']}
    for i,r in enumerate(rows):
        for k,m in maps.items():
            v=r[k]
            if v is not None:
                if v in m: uf.join(i,m[v])
                else: m[v]=i
    near=[]
    hashes=[int(r['dhash'],16) for r in rows]
    thumbnail_array=np.stack(thumbnails)
    for i in range(len(rows)):
        pixel_distances=np.abs(thumbnail_array[:i]-thumbnail_array[i]).mean(axis=(1,2,3)) if i else []
        for j in range(i):
            distance=(hashes[i]^hashes[j]).bit_count()
            if distance<=4 or pixel_distances[j]<=.025:
                near.append({'a':rows[j]['image_id'],'b':rows[i]['image_id'],'hamming':distance,'rgb_mae_32':float(pixel_distances[j]),
                             'action':'conservative leakage-group union; NOT identity label'})
                uf.join(i,j)
    components=defaultdict(list)
    for i in range(len(rows)): components[uf.find(i)].append(i)
    for inds in components.values():
        group=object_hash(sorted(rows[i]['image_id'] for i in inds))[:16]
        for i in inds: rows[i]['duplicate_group']=group
    # Remove only exact pixel duplicates. Keep lineage/noise variants for audit but select one per group later.
    seen={}
    for r in rows:
        if r['pixel_sha256'] in seen: r['excluded_reason']='exact_pixel_duplicate_of:'+seen[r['pixel_sha256']]
        else: seen[r['pixel_sha256']]=r['image_id']
    write_json(out, rows)
    summary={'images':len(rows),'corrupt':len(corrupt),'exact_pixel_duplicates':sum(bool(r['excluded_reason'] and r['excluded_reason'].startswith('exact_pixel_duplicate')) for r in rows),
             'sources':dict(Counter(r['source'] for r in rows)), 'categories_not_designs':dict(Counter(r['category'] for r in rows)),
             'sizes':dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
             'original_groups':len(set(r['original_group'] for r in rows)), 'leakage_groups':len(components),
             'near_duplicate_candidate_pairs':len(near), 'reliable_designs':len(set(r['design_id'] for r in rows if r['design_id'])),
             'manifest_sha256':digest(out),'method':'dHash<=4 OR bilinear 32x32 RGB MAE<=0.025 conservatively groups; neither is identity truth'}
    write_json(Path(out).with_name('audit_summary.json'),summary)
    write_json(Path(out).with_name('audit_evidence.json'),{'corrupt':corrupt,'near_candidates':near})
    return summary


def make_splits(manifest, out, mode='instance', seed=17, max_per_split=None):
    """Freeze groups before augmentation. Natural requires trusted design/capture metadata."""
    if Path(out).exists(): raise FileExistsError('Frozen split exists; use a new output path for a revised protocol')
    rows=read_json(manifest)
    uf=UnionFind(len(rows)); seen={}
    for i,r in enumerate(rows):
        for key in ('duplicate_group','original_group','capture_group','design_id'):
            if r.get(key):
                v=(key,r[key])
                if v in seen: uf.join(i,seen[v])
                else: seen[v]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[uf.find(i)].append(r)
    keys=sorted(groups,key=lambda k:min(r['image_id'] for r in groups[k]))
    rng=random.Random(seed);rng.shuffle(keys)
    if len(keys)<7: raise ValueError('Need at least 7 independent groups for nonempty partitions')
    nv=max(1,round(len(keys)*.15)); nt=nv
    partitions={'val':keys[:nv], 'test':keys[nv:nv+nt], 'train':keys[nv+nt:]}
    for split,ks in partitions.items():
        selected=0
        for k in ks:
            candidates=sorted(groups[k],key=lambda r:r['image_id'])
            used=set()
            for r in candidates:
                r['split']=split; r['role']='excluded'
                if r['excluded_reason']: continue
                if mode=='instance':
                    if used or (max_per_split and selected>=max_per_split):
                        r['excluded_reason']='conservative_one_representative_per_leakage_group_or_budget'
                        continue
                    r['role']='train' if split=='train' else 'synthetic_gallery'
                    used.add(k);selected+=1
                elif mode=='supervised':
                    if not r['design_id'] or r['annotation_confidence']!='high' or not r.get('capture_group'):
                        r['excluded_reason']='natural_protocol_requires_trusted_design_and_capture';continue
                    r['role']='train' if split=='train' else 'candidate'
                else: raise ValueError('Unknown split mode')
    if mode=='supervised':
        for split in ('val','test'):
            designs=defaultdict(list)
            for r in rows:
                if r['split']==split and r['role']=='candidate': designs[r['design_id']].append(r)
            for design,rs in designs.items():
                rs.sort(key=lambda r:r['image_id']); gallery=rs[0]
                gallery['role']='gallery'
                for r in rs[1:]:
                    if r['duplicate_group']==gallery['duplicate_group'] or r['capture_group']==gallery['capture_group']:
                        r['role']='excluded';r['excluded_reason']='same_or_near_capture_as_gallery'
                    else: r['role']='query'
                if not any(r['role']=='query' for r in rs):
                    gallery['role']='excluded';gallery['excluded_reason']='no_independent_query'
        for split in ('train','val','test'):
            if not any(r['split']==split and r['role']!='excluded' for r in rows):
                raise ValueError(f'Insufficient trusted identity/capture coverage for {split}')
    result={'version':1,'seed':seed,'mode':mode,'manifest_sha256':digest(manifest),
            'grouping':'union duplicate/original/capture and supported design; ~70/15/15 groups',
            'max_per_split':max_per_split,'rows':rows,'protocols':{}}
    for split in ('val','test'):
        selected=[r for r in rows if r['split']==split and r['role']!='excluded']
        if mode=='instance':
            g=[r['image_id'] for r in selected];q=g[:]
            # Fixed positive synthetic views and all ordered other-instance pairs; not verified different designs.
            pairs=[{'a':a,'b':b,'label':int(a==b)} for a in q for b in g]
            name='synthetic_instance_recoloring'
        else:
            g=[r['image_id'] for r in selected if r['role']=='gallery']
            q=[r['image_id'] for r in selected if r['role']=='query']
            lookup={r['image_id']:r for r in selected}
            pairs=[{'a':a,'b':b,'label':int(lookup[a]['design_id']==lookup[b]['design_id'])} for a in q for b in g]
            name='natural_distinct_capture'
        result['protocols'][split]={'name':name,'gallery':g,'query':q,'pairs':pairs,'recolor_seed':seed}
    assert_no_leakage(result)
    write_json(out,result)
    return {'split_sha256':digest(out),'counts':dict(Counter((r['split']+'/'+r['role']) for r in rows))}


def assert_no_leakage(spec):
    for key in ('original_group','capture_group','duplicate_group','design_id'):
        seen={}
        for r in spec['rows']:
            v=r.get(key)
            if not v: continue
            if v in seen and seen[v]!=r['split']: raise ValueError(f'Outer leakage: {key}')
            seen[v]=r['split']
    if spec['mode']=='supervised':
        for part in ('val','test'):
            gallery=[r for r in spec['rows'] if r['split']==part and r['role']=='gallery']
            query=[r for r in spec['rows'] if r['split']==part and r['role']=='query']
            for key in ('original_group','capture_group','duplicate_group'):
                if {r[key] for r in gallery}&{r[key] for r in query}: raise ValueError(f'Gallery/query leakage: {key}')


def verify_files(rows, root):
    """Reject changed source bytes when consuming a frozen manifest."""
    from .common import safe_path
    for r in rows:
        if digest(safe_path(root,r['relative_path']))!=r['sha256']:
            raise ValueError(f"Source bytes changed for opaque image {r['image_id']}; re-audit and define a new experiment")


def policy_annotations(root, policy_file, exclude_vendor=True):
    """Reproducible public-source exclusions without publishing source filenames."""
    policy=read_json(policy_file);root=Path(root);annotations={}
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'):continue
        rel=p.relative_to(root).as_posix()
        if exclude_vendor and rel.startswith('vendor/'):
            annotations[rel]=dict(policy['vendor_annotation'])
        elif digest(p) in policy['excluded_public_file_sha256']:
            annotations[rel]=dict(policy['public_exclusion_annotation'])
    return annotations

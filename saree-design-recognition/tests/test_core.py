import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from saree.common import seed_all,write_json,read_json
from saree.data import audit,make_splits,assert_no_leakage
from saree.fixtures import generate
from saree.losses import supervised_contrastive
from saree.metrics import retrieval_metrics,select_threshold,verification_metrics
from saree.model import Embedder,save_model,load_model
from saree.sampling import PKSampler
from saree.transforms import Preprocess,fit,recolor
from saree.retrieval import build_gallery,load_gallery,retrieve,verify


@pytest.fixture(autouse=True)
def seeded(): seed_all(17)


def test_loss_gradients_and_no_positive():
    x=torch.randn(6,8,requires_grad=True);z=torch.nn.functional.normalize(x,dim=1)
    loss=supervised_contrastive(z,torch.tensor([0,0,1,1,2,2]));loss.backward()
    assert torch.isfinite(loss) and x.grad.norm()>0
    with pytest.raises(ValueError,match='no valid positive'):supervised_contrastive(z,torch.arange(6))


def test_model_roundtrip(tmp_path):
    model=Embedder(pretrained=False).eval();x=torch.randn(2,3,224,224)
    with torch.no_grad(): before=model(x)
    assert torch.allclose(before.norm(dim=1),torch.ones(2),atol=1e-6)
    save_model(tmp_path/'m.pt',model,{'mode':'synthetic_smoke'})
    loaded,_=load_model(tmp_path/'m.pt')
    with torch.no_grad():after=loaded(x);again=loaded(x)
    assert torch.equal(before,after) and torch.equal(after,again)


def test_baselines_have_no_projection():
    for kind in ('rgb','gray'):
        model=Embedder(kind,pretrained=False)
        assert isinstance(model.head,torch.nn.Identity) and model.dim==576


def test_metrics_hand_computed():
    # Relevant at rank 1 and rank 3: AP=(1+2/3)/2=5/6.
    r=retrieval_metrics([[.9,.8,.7]],['a'],['a','b','a'])
    assert r['mAP']==pytest.approx(5/6) and r['recall_at_1']==1 and r['recall_at_5'] is None
    r=retrieval_metrics([[.9,.8]],['missing'],['a','b'])
    assert r['mAP'] is None and r['queries_without_positive']==1


def test_validation_threshold_only():
    with pytest.raises(ValueError):select_threshold([0,1],[.2,.8],'test')
    cal=select_threshold([0,0,1,1],[.1,.2,.8,.9],'val')
    assert cal['threshold']==.8
    assert verification_metrics([0,0,1,1],[.1,.2,.8,.9],cal['threshold'])['balanced_accuracy']==1
    assert not verification_metrics([1],[.9],.5)['available']


def test_sampler_distinct_captures():
    rows=[{'design_id':str(d),'capture_group':f'{d}_{k}','original_group':f'{d}_{k}'} for d in range(4) for k in range(2)]
    sampler=PKSampler(rows,p=4,k=2)
    batch=next(sampler.batches());assert len(batch)==8 and len(set(batch))==8
    assert batch==next(sampler.batches())
    with pytest.raises(ValueError):PKSampler(rows[:2],p=4,k=2)


def test_audit_duplicate_and_split(tmp_path):
    root=tmp_path/'data';generate(root)
    (root/'copy.png').write_bytes((root/'fixture_000.png').read_bytes())
    (root/'corrupt.jpg').write_bytes(b'not an image')
    summary=audit(root,tmp_path/'manifest.json')
    assert summary['corrupt']==1 and summary['exact_pixel_duplicates']==1
    make_splits(tmp_path/'manifest.json',tmp_path/'split.json')
    spec=read_json(tmp_path/'split.json');assert_no_leakage(spec)
    duplicates=[r for r in spec['rows'] if r['sha256']==spec['rows'][0]['sha256']]
    assert len(set(r['split'] for r in duplicates))==1
    broken=copy.deepcopy(spec);broken['rows'][0]['split']='intruder'
    with pytest.raises(ValueError):assert_no_leakage(broken)
    with pytest.raises(FileExistsError):make_splits(tmp_path/'manifest.json',tmp_path/'split.json')


def test_missing_data_never_silent_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):audit(tmp_path/'missing',tmp_path/'m.json')
    with pytest.raises(ValueError,match='No readable'):audit(tmp_path,tmp_path/'m.json')
    with pytest.raises(FileNotFoundError):load_model(tmp_path/'missing.pt')


def test_transforms_preserve_structure():
    im=Image.new('RGB',(100,200),(255,0,0));canvas=fit(im)
    assert canvas.size==(224,224)
    assert canvas.getpixel((0,100))==(124,116,104)
    assert canvas.getpixel((112,100))==(255,0,0)
    assert torch.equal(Preprocess()(im),Preprocess()(im))
    assert np.array_equal(np.sort(np.asarray(im),axis=2),np.sort(np.asarray(recolor(im,17)),axis=2))


def test_cache_rejection(tmp_path):
    root=tmp_path/'data';generate(root);audit(root,tmp_path/'m.json');make_splits(tmp_path/'m.json',tmp_path/'s.json')
    model=Embedder(pretrained=False).eval();save_model(tmp_path/'ck.pt',model,{'mode':'synthetic_smoke'})
    build_gallery(tmp_path/'ck.pt',tmp_path/'s.json',root,tmp_path/'gallery')
    z,meta=load_gallery(tmp_path/'ck.pt',tmp_path/'gallery',tmp_path/'s.json')
    assert z.shape[1]==256
    result=retrieve(tmp_path/'ck.pt',tmp_path/'gallery',tmp_path/'s.json',root/'fixture_000.png')
    assert 'cosine_similarity' in result[0] and 'confidence' not in result[0]
    save_model(tmp_path/'changed.pt',Embedder(pretrained=False),{'mode':'synthetic_smoke'})
    with pytest.raises(ValueError,match='Incompatible'):load_gallery(tmp_path/'changed.pt',tmp_path/'gallery',tmp_path/'s.json')
    a=read_json(tmp_path/'s.json');a['seed']=99;write_json(tmp_path/'other.json',a)
    with pytest.raises(ValueError,match='split'):load_gallery(tmp_path/'ck.pt',tmp_path/'gallery',tmp_path/'other.json')


def test_supervised_split_requires_truth(tmp_path):
    root=tmp_path/'data';generate(root);audit(root,tmp_path/'m.json')
    with pytest.raises(ValueError,match='Insufficient trusted'):make_splits(tmp_path/'m.json',tmp_path/'s.json','supervised')


def test_exif_orientation(tmp_path):
    from saree.transforms import load_image
    im=Image.new('RGB',(100,200),(20,40,60));exif=Image.Exif();exif[274]=6
    im.save(tmp_path/'rotated.jpg',exif=exif)
    assert load_image(tmp_path/'rotated.jpg').size==(200,100)


def test_natural_outer_and_gallery_capture_guards():
    rows=[dict(image_id='a',split='val',role='gallery',design_id='d',original_group='oa',capture_group='ca',duplicate_group='ga'),
          dict(image_id='b',split='val',role='query',design_id='d',original_group='ob',capture_group='cb',duplicate_group='gb')]
    spec={'mode':'supervised','rows':rows};assert_no_leakage(spec)
    spec['rows'][1]['capture_group']='ca'
    with pytest.raises(ValueError,match='Gallery/query leakage'):assert_no_leakage(spec)
    spec['rows'][1]['capture_group']='cb';spec['rows'][1]['split']='test'
    with pytest.raises(ValueError,match='Outer leakage'):assert_no_leakage(spec)


def test_test_partition_one_use(tmp_path):
    from saree.evaluation import evaluate
    from saree.common import digest
    write_json(tmp_path/'split.json',{'mode':'instance','rows':[]})
    write_json(tmp_path/'lock.json',{'split_sha256':digest(tmp_path/'split.json')})
    out=tmp_path/'out';out.mkdir();write_json(out/'TEST_STARTED.json',{'status':'started'})
    with pytest.raises(FileExistsError,match='already started'):evaluate(tmp_path/'lock.json',tmp_path/'split.json',tmp_path,out)


def test_annotation_requires_evidence(tmp_path):
    root=tmp_path/'data';root.mkdir();Image.new('RGB',(20,20)).save(root/'x.png')
    write_json(tmp_path/'ann.json',{'x.png':{'design_id':'invented'}})
    with pytest.raises(ValueError,match='provenance'):audit(root,tmp_path/'manifest.json',tmp_path/'ann.json')


def test_frozen_source_bytes_rejected(tmp_path):
    from saree.data import verify_files
    root=tmp_path/'data';root.mkdir();Image.new('RGB',(20,20),'red').save(root/'x.png')
    audit(root,tmp_path/'m.json');rows=read_json(tmp_path/'m.json');verify_files(rows,root)
    Image.new('RGB',(20,20),'blue').save(root/'x.png')
    with pytest.raises(ValueError,match='Source bytes changed'):verify_files(rows,root)


def test_pixel_screen_catches_dhash_instability(tmp_path):
    root=tmp_path/'data';root.mkdir()
    a=np.full((64,64,3),128,dtype=np.uint8)
    b=a.copy();b[:,32:,:]+=2
    Image.fromarray(a).save(root/'a.png');Image.fromarray(b).save(root/'b.png')
    audit(root,tmp_path/'m.json');rows=read_json(tmp_path/'m.json')
    assert (int(rows[0]['dhash'],16)^int(rows[1]['dhash'],16)).bit_count()>4
    assert rows[0]['duplicate_group']==rows[1]['duplicate_group']
    assert all(r['design_id'] is None for r in rows)

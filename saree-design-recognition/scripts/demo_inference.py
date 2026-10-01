"""Exercise documented CLI inference on existing gallery files; not evaluation."""
import argparse,json,subprocess,sys
from pathlib import Path
from saree.common import read_json,write_json
p=argparse.ArgumentParser();p.add_argument('--run-root',default='private/run_v2');p.add_argument('--split',default='private/split_v2.json');p.add_argument('--data-root',default='private/data');a=p.parse_args()
root=Path(a.run_root);spec=read_json(a.split);lookup={r['image_id']:r for r in spec['rows']};ids=spec['protocols']['test']['gallery']
if len(ids)<2:raise ValueError('Need at least two gallery images for API demo')
images=[str(Path(a.data_root)/lookup[i]['relative_path']) for i in ids[:2]]
base=[sys.executable,'-m','saree.cli','--device','cpu']
ck=str(root/'trained/best.pt');cal=str(root/'trained/calibration.json')
r=subprocess.run(base+['retrieve','--checkpoint',ck,'--split',a.split,'--gallery',str(root/'gallery'),'--image',images[0],'--k','5'],check=True,capture_output=True,text=True)
v=subprocess.run(base+['verify','--checkpoint',ck,'--calibration',cal,'--a',images[0],'--b',images[1]],check=True,capture_output=True,text=True)
write_json(root/'demo_api_results.private.json',{'retrieval':json.loads(r.stdout),'verification':json.loads(v.stdout),'purpose':'API sanity only; query taken from gallery; no new evaluation metric'})
print(json.dumps({'retrieve_cli_exit':r.returncode,'verify_cli_exit':v.returncode,'top_k_returned':len(json.loads(r.stdout)),'calibration_claim_supported':json.loads(v.stdout)['claim_supported'],'purpose':'API sanity, not evaluation'},indent=2))

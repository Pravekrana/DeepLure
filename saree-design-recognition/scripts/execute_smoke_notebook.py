"""Execute submission notebook in explicit synthetic mode; never access real test data."""
import json,os,sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
kernel=root/'private/jupyter/kernels/deeplure';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'DeepLure local','language':'python'}))
os.environ['JUPYTER_PATH']=str(root/'private/jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(root/'private/jupyter-runtime')
os.environ['IPYTHONDIR']=str(root/'private/ipython')
nb=nbformat.read(root/'notebooks/DeepLure_Submission.ipynb',as_version=4)
for cell in nb.cells:
    if cell.cell_type=='code':
        cell.source=cell.source.replace('MODE = "real"','MODE = "synthetic_smoke"').replace('private/notebook_run','private/notebook_smoke_v2')
nb.cells.insert(1,nbformat.v4.new_markdown_cell('**EXECUTED SOFTWARE SMOKE ONLY.** Generated geometric images; pretrained initialization. No natural saree recognition or independent validation is established by this notebook.'))
NotebookClient(nb,timeout=600,kernel_name='deeplure',resources={'metadata':{'path':str(root)}}).execute()
nb.metadata['execution_scope']='synthetic software smoke only; no real test used'
nbformat.write(nb,root/'notebooks/DeepLure_Smoke_Executed.ipynb')
print('Executed all code cells in explicit synthetic mode.')

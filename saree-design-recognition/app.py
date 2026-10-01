"""Local-only UI. Launch with --server.address 127.0.0.1."""
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
from saree.common import read_json,digest
from saree.model import load_model
from saree.retrieval import load_gallery,encode,verify

st.set_page_config(page_title='Saree design retrieval',layout='centered')
st.title('Saree design retrieval')
st.caption('Local uploads • cosine similarity is not a probability')
ck=st.sidebar.text_input('Checkpoint','private/run_v2/trained/best.pt')
split=st.sidebar.text_input('Frozen split','private/split_v2.json')
gallery=st.sidebar.text_input('Gallery directory','private/run_v2/gallery')
cal=st.sidebar.text_input('Validation calibration','private/run_v2/trained/calibration.json')
missing=[x for x in [ck,split,cal,str(Path(gallery)/'cache.json'),str(Path(gallery)/'embeddings.npy')] if not Path(x).is_file()]
if missing:
    st.error('Missing assets: '+', '.join(missing));st.stop()

@st.cache_resource
def assets(checkpoint,folder,split_file,ck_hash,split_hash,cache_hash,embedding_hash):
    model,meta=load_model(checkpoint);z,cache=load_gallery(checkpoint,folder,split_file)
    return model,meta,z,cache

try:
    model,meta,g,cache=assets(ck,gallery,split,digest(ck),digest(split),digest(Path(gallery)/'cache.json'),digest(Path(gallery)/'embeddings.npy'))
    calibration=read_json(cal)
    if calibration['checkpoint_sha256']!=digest(ck) or calibration['split_sha256']!=digest(split): raise ValueError('Calibration/checkpoint/split mismatch')
except Exception as exc:
    st.error(f'Asset error: {exc}');st.stop()
if cache['protocol']!='natural_distinct_capture':
    st.warning('Experimental instance model. Threshold calibrated on synthetic recoloring, not independent same-design captures. Natural design-recognition performance is unavailable.')
with st.expander('Model and metric provenance'):
    st.json({'checkpoint_sha256':digest(ck),'metadata':meta,'protocol':cache['protocol'],'validation_retrieval':calibration['retrieval'],'threshold_rule':calibration['rule']})
t1,t2=st.tabs(['Identify','Verify pair'])
with t1:
    query=st.file_uploader('RGB query image',type=['jpg','jpeg','png'],key='q')
    k=st.slider('Top matches',1,min(20,len(g)),min(5,len(g)))
    if query:
        im=Image.open(query);st.image(im,width=260)
        q=encode(model,[im]);scores=(q@g.T)[0];order=np.argsort(-scores,kind='stable')[:k]
        st.dataframe([{'opaque_image_id':cache['ids'][i],'design_id':cache['design_ids'][i] or 'unknown','cosine_similarity':float(scores[i])} for i in order])
        st.caption('Image-level ranking. No open-set acceptance threshold is applied.')
with t2:
    a=st.file_uploader('First image',type=['jpg','jpeg','png'],key='a')
    b=st.file_uploader('Second image',type=['jpg','jpeg','png'],key='b')
    if a and b:
        result=verify(ck,cal,Image.open(a),Image.open(b));st.json(result)
        st.caption('The decision uses the saved validation threshold unchanged. Synthetic calibration does not establish genuine design identity.')

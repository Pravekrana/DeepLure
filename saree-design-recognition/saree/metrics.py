import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, balanced_accuracy_score


def retrieval_metrics(scores, query_labels, gallery_labels):
    scores=np.asarray(scores); gl=np.asarray(gallery_labels); aps=[];hits1=[];hits5=[];missing=0;positives=[]
    for row,label in zip(scores,query_labels):
        relevant=gl==label; count=int(relevant.sum());positives.append(count)
        if not count: missing+=1;continue
        order=np.argsort(-row,kind='stable'); rel=relevant[order]
        aps.append(float(((np.cumsum(rel)/np.arange(1,len(rel)+1))*rel).sum()/count))
        hits1.append(bool(rel[0]));hits5.append(bool(rel[:5].any()))
    return {'recall_at_1':float(np.mean(hits1)) if aps else None,
            'recall_at_5':float(np.mean(hits5)) if aps and len(gl)>=5 else None,
            'mAP':float(np.mean(aps)) if aps else None,'queries':len(query_labels),
            'eligible_queries':len(aps),'queries_without_positive':missing,'gallery_images':len(gl),
            'gallery_identities':len(set(gallery_labels)),'positives_per_query':positives,'ranking_unit':'image'}


def select_threshold(labels, scores, split):
    if split!='val': raise ValueError('Threshold selection is permitted only on validation')
    labels=np.asarray(labels);scores=np.asarray(scores)
    if len(np.unique(labels))!=2: raise ValueError('Threshold calibration needs both classes')
    thresholds=np.r_[np.nextafter(scores.max(),np.inf),np.unique(scores)]
    # Maximum balanced accuracy; highest threshold breaks ties (more conservative acceptance).
    best=max(((float(balanced_accuracy_score(labels,scores>=t)),float(t)) for t in thresholds))
    return {'threshold':best[1],'validation_balanced_accuracy':best[0],
            'rule':'maximize validation balanced accuracy; highest threshold on ties','selected_on':'val'}


def verification_metrics(labels,scores,threshold):
    labels=np.asarray(labels);scores=np.asarray(scores)
    npos=int(labels.sum());nneg=len(labels)-npos
    if not npos or not nneg:
        return {'available':False,'reason':'Both pair classes required','positive_pairs':npos,'negative_pairs':nneg}
    p,r,f,_=precision_recall_fscore_support(labels,scores>=threshold,average='binary',zero_division=0)
    fpr,tpr,ts=roc_curve(labels,scores)
    return {'available':True,'roc_auc':float(roc_auc_score(labels,scores)), 'precision':float(p),
            'recall':float(r),'f1':float(f),'balanced_accuracy':float(balanced_accuracy_score(labels,scores>=threshold)),
            'threshold':float(threshold),'positive_pairs':npos,'negative_pairs':nneg,
            'positive_fraction':npos/len(labels),'negative_count_fpr_resolution':1/nneg,
            'roc':{'fpr':fpr.tolist(),'tpr':tpr.tolist()},
            'low_fpr_claim':None,'dependence':'pairs share queries/gallery; not independent trials'}

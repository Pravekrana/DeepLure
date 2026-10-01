"""Copy aggregate evidence only and derive reports from completed experiment outputs."""
from pathlib import Path
from collections import Counter
import json,shutil
from saree.common import read_json,write_json,digest
root=Path('private/run_v2');reports=Path('reports');m=read_json(root/'evaluation/metrics.json');spec=read_json('private/split_v2.json')
for name in ['metrics.json','results.csv','roc.png']:
    shutil.copyfile(root/'evaluation'/name,reports/name)
for name in ('trained','no_color'):
    shutil.copyfile(root/name/'training_log.json',reports/f'{name}_training_log.json')
    shutil.copyfile(root/name/'training_summary.json',reports/f'{name}_training_summary.json')
for name in ('trained','no_color','rgb','gray'):
    shutil.copyfile(root/name/'calibration.json',reports/f'{name}_validation.json')
audit=read_json('private/audit_summary.json');rows=read_json('private/manifest_v2.json');lookup={r['image_id']:r for r in rows}
evidence=read_json('private/audit_evidence.json')
audit['cross_source_near_duplicate_candidate_pairs']=sum(lookup[e['a']]['source']!=lookup[e['b']]['source'] for e in evidence['near_candidates'])
audit['exif_orientation_counts']=dict(Counter(str(r['exif_orientation']) for r in rows))
write_json(reports/'audit_summary.json',audit)
write_json(reports/'split_summary.json',{'seed':spec['seed'],'mode':spec['mode'],'manifest_sha256':spec['manifest_sha256'],'split_sha256':digest('private/split_v2.json'),
    'counts':dict(Counter(r['split']+'/'+r['role'] for r in spec['rows'])),'rank_unit':'image',
    'positive_semantics':'same photograph after deterministic synthetic transformation','negative_semantics':'different representative instance, not verified different design',
    'pair_counts':{k:{'positive':sum(p['label'] for p in v['pairs']),'negative':sum(not p['label'] for p in v['pairs']),'gallery':len(v['gallery']),'query':len(v['query'])} for k,v in spec['protocols'].items()}})
lines=['# Evaluation: executed evidence and limits','',
'**Natural same-design verification and real cross-colorway retrieval are unavailable.** No reliable design/capture/colorway ground truth was established. These results measure controlled transformations of the same photographs, using real public images for training and held-out synthetic testing. They are not an unseen-design benchmark. The initial experiment was invalidated after a duplicate-audit correction; these are corrected v2 results with prior corpus/test exposure disclosed in CORRECTION_LOG.md.','',
'## Data, freeze and coverage','',
'Acquired 1,468 Kaggle images and 165 vendor images; all 1,633 decoded successfully. Five exact pixel duplicates were identified. Vendor images were AI-inspected and excluded because repeated/cropped-view independence and colorways were unresolved. The category folders are not design identities. Conservative derivative/duplicate grouping left 211 train, 46 validation and 45 test representatives. All 302 representatives used here are from Kaggle. Other images remain in the manifest with exclusions. No reliable natural designs or natural pairs are claimed.','',
'Outer splits use seed 17 and unions of documented lineage, duplicate candidates, capture groups and supported identities, before augmentations. This prevents detected original/capture leakage; unknown exact-design overlap remains possible. Source filename collisions and dHash false positives can overmerge groups; missed similarities can remain. No model clusters supply labels. Frozen split and source hashes are in split_summary.json.','',
'Validation has 46 positive and 2,070 ordered other-instance negative pairs. Test has 45 positives and 1,980 negatives (2,025 total, 2.22% positive) per stress. Every query has one eligible gallery positive; zero queries lack an eligible positive. The gallery has 45 images/instances, with true design count unknown. No low-FPR or independent-pair reliability claim is made.','',
'## Models and selection','',
'Both trained variants executed four CPU epochs with batch size 12 originals, two views per original, AdamW, learning rate 0.0003, weight decay 0.0001, temperature 0.1, gradient clipping 5, and seed 17. Eighteen batches ran per epoch. Epoch 1 freezes the backbone; epochs 2–4 attempt fine-tuning with fixed batch-normalization statistics. Validation mAP chooses the earliest maximum. Both selected epoch 1: the deployed trained model therefore has a learned 256-D head on the pretrained backbone, not a validated benefit from backbone fine-tuning. The no-color ablation changes only the color-augmentation flag. Skipping color operations changes RNG consumption, so individual geometry draws are not bit-matched; their distribution and sampling seed are unchanged. This is a single-seed comparison. Full logs include losses, gradients, embedding dispersion and durations.','',
'Frozen RGB/gray baselines use the actual 576-D pooled ImageNet features with no random projection. The same frozen partitions, image-level ranking, stable tie rule and pair lists apply to all models. No architecture decision used test results. All model and calibration hashes were locked before one final test invocation; the earlier v1 test was invalidated after duplicate-screening misses. The corrected v2 protocol was evaluated once after locking, but prior corpus exposure means it is not a pristine untouched holdout. See CORRECTION_LOG.md.','',
'## Controlled transformations','',
'Primary recoloring permutes RGB channels, selecting one of three nonidentity permutations using seed 17 plus query index. Secondary grayscale removes chroma. Secondary inversion maps RGB to 255−RGB, preserving boundaries while reversing contrast polarity. Inversion was added before final locking because validation on channel permutation saturated. All are deterministic derivatives of the gallery photograph; none is a new capture or real colorway. Each model uses its primary validation-recolor threshold unchanged across secondary stresses.','',
'## Retrieval results','',
'Ranks are images, not aggregated designs. Recall@K is the fraction of eligible queries with a positive in the first K ranks. AP averages precision at positive ranks divided by all eligible positives; mAP averages AP over eligible queries. With one positive per query here, AP is reciprocal rank.','',
'| Model | Stress | Recall@1 | Recall@5 | mAP |','|---|---|---:|---:|---:|']
for name,r in m['models'].items():
    for stress in ('recolor','grayscale','inversion'):
        a=r[stress]['retrieval'];lines.append(f"| {name} | {stress} | {a['recall_at_1']:.4f} | {a['recall_at_5']:.4f} | {a['mAP']:.4f} |")
lines += ['','## Verification results','','Threshold maximizes validation balanced accuracy; highest threshold breaks ties. Scores are cosine similarities, not probabilities. These synthetic thresholds do not establish natural same-design decisions and are not open-set gallery thresholds.','','| Model | Stress | ROC-AUC | Precision | Recall | F1 | Balanced accuracy | Threshold |','|---|---|---:|---:|---:|---:|---:|---:|']
for name,r in m['models'].items():
    for stress in ('recolor','grayscale','inversion'):
        b=r[stress]['verification'];lines.append(f"| {name} | {stress} | {b['roc_auc']:.4f} | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} | {b['balanced_accuracy']:.4f} | {b['threshold']:.6f} |")
lines += ['','## Interpretation and error counts','','Channel permutation retains exact photographic geometry and many texture cues; strong results can reflect shortcuts rather than design recognition. The measured tables, not the proposed architecture, determine comparative claims. No superiority on real designs is established. Frozen grayscale is a serious baseline, but even grayscale is not invariant to all palette luminances or contrast polarity.','','| Model | Recolor top-1 errors | Grayscale top-1 errors | Inversion top-1 errors |','|---|---:|---:|---:|']
for name,r in m['models'].items():
    counts=[len(read_json(root/'evaluation'/f'{name}_{stress}_errors.private.json')) for stress in ('recolor','grayscale','inversion')]
    lines.append('| '+name+' | '+' | '.join(map(str,counts))+' |')
lines += ['','Private score matrices and opaque-ID error lists support inspection without sharing images. No uncertainty interval was estimated; image-sharing pair dependence prevents treating 2,025 pairs as independent examples. Hard different-design/similar-palette performance is unavailable without reliable design truth. The synthetic geometric software fixtures use identical palettes but are not dataset evidence.','',
'## Executed versus not executed','','Executed: metadata/image audit, frozen group split, four-epoch real-image instance training and ablation, frozen baselines, validation calibration, locked final synthetic evaluation, checkpoint checks, gallery/inference, CPU efficiency, tests and explicit synthetic notebook execution. Not executed/supported: real identity-supervised training, natural independent-capture design metrics, real alternate-colorway metrics, CUDA/MPS runs, independent human validation, publication or form submission. See SOURCES.md and AI_DISCLOSURE.md.']
from saree.model import load_model
selected_epochs={name:load_model(root/name/'best.pt')[1]['epoch'] for name in ('trained','no_color')}
text='\n'.join(lines)+'\n'
text=text.replace('Both selected epoch 1: the deployed trained model therefore has a learned 256-D head on the pretrained backbone, not a validated benefit from backbone fine-tuning.', f"Selected epochs: color-augmented={selected_epochs['trained']}, no-color={selected_epochs['no_color']}. Epoch 1 is head-only; a selected later epoch includes backbone fine-tuning. Selection follows validation, not final test performance.")
(reports/'EVALUATION.md').write_text(text)
eff=read_json(root/'efficiency_final.json');write_json(reports/'efficiency.json',eff)
(reports/'EFFICIENCY.md').write_text(f'''# Measured efficiency
Selected 256-D model: **{eff['parameters']:,} parameters**, {eff['embedding_bytes']} bytes per float32 embedding. Checkpoint is {eff['checkpoint_bytes']:,} bytes and includes training optimizer state; it is the exact artifact used by inference. Frozen baselines output 576 float32 values (2,304 bytes/image) with 927,008 backbone parameters.

Device: {eff['device']}, {eff['hardware']}; {eff['platform']}. Input {eff['input_shape']}, batch {eff['batch_size']}, {eff['precision']}, {eff['threads']} Torch threads. {eff['warmup']} warm-up calls and {eff['repeats']} timed repeats; final benchmark run after training/evaluation stopped. CUDA/MPS code synchronizes when selected; this CPU measurement needs no accelerator synchronization.

| Operation | Median ms | p95 ms |
|---|---:|---:|
| Encoder only | {eff['encoder']['median_ms']:.3f} | {eff['encoder']['p95_ms']:.3f} |
| Preprocessing, decoded 640×480 PIL image | {eff['preprocessing_excludes_disk_decode']['median_ms']:.3f} | {eff['preprocessing_excludes_disk_decode']['p95_ms']:.3f} |
| Exact CPU search + sorting, 1,000 generated vectors | {eff['search']['median_ms']:.3f} | {eff['search']['p95_ms']:.3f} |

The actual test gallery contains 45 images; the 1,000-vector search benchmark is explicitly a generated scaling probe. Preprocessing excludes disk read/decode, and timings exclude checkpoint loading and app overhead. No local-patch/multiview cost is hidden. Latency varies with CPU scheduling and WSL; these are local measurements, not device-independent promises. FLOPs/MACs were not measured and are unavailable.
''')
write_json(reports/'artifact_hashes.json',{'pretrained_checkpoint_sha256':digest('private/torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'),
    'split_sha256':digest('private/split_v2.json'),'models':{name:digest(root/name/('best.pt' if name in ('trained','no_color') else 'baseline.pt')) for name in ('trained','no_color','rgb','gray')},
    'approach_characters':len(Path('APPROACH_500_CHARS.txt').read_text())})
print('Aggregate reports generated from actual outputs')

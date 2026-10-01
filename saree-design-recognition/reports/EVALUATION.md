# Evaluation: executed evidence and limits

**Natural same-design verification and real cross-colorway retrieval are unavailable.** No reliable design/capture/colorway ground truth was established. These results measure controlled transformations of the same photographs, using real public images for training and held-out synthetic testing. They are not an unseen-design benchmark. The initial experiment was invalidated after a duplicate-audit correction; these are corrected v2 results with prior corpus/test exposure disclosed in CORRECTION_LOG.md.

## Data, freeze and coverage

Acquired 1,468 Kaggle images and 165 vendor images; all 1,633 decoded successfully. Five exact pixel duplicates were identified. Vendor images were AI-inspected and excluded because repeated/cropped-view independence and colorways were unresolved. The category folders are not design identities. Conservative derivative/duplicate grouping left 211 train, 46 validation and 45 test representatives. All 302 representatives used here are from Kaggle. Other images remain in the manifest with exclusions. No reliable natural designs or natural pairs are claimed.

Outer splits use seed 17 and unions of documented lineage, duplicate candidates, capture groups and supported identities, before augmentations. This prevents detected original/capture leakage; unknown exact-design overlap remains possible. Source filename collisions and dHash false positives can overmerge groups; missed similarities can remain. No model clusters supply labels. Frozen split and source hashes are in split_summary.json.

Validation has 46 positive and 2,070 ordered other-instance negative pairs. Test has 45 positives and 1,980 negatives (2,025 total, 2.22% positive) per stress. Every query has one eligible gallery positive; zero queries lack an eligible positive. The gallery has 45 images/instances, with true design count unknown. No low-FPR or independent-pair reliability claim is made.

## Models and selection

Both trained variants executed four CPU epochs with batch size 12 originals, two views per original, AdamW, learning rate 0.0003, weight decay 0.0001, temperature 0.1, gradient clipping 5, and seed 17. Eighteen batches ran per epoch. Epoch 1 freezes the backbone; epochs 2–4 attempt fine-tuning with fixed batch-normalization statistics. Validation mAP chooses the earliest maximum. Selected epochs: color-augmented=2, no-color=1. Epoch 1 is head-only; a selected later epoch includes backbone fine-tuning. Selection follows validation, not final test performance. The no-color ablation changes only the color-augmentation flag. Skipping color operations changes RNG consumption, so individual geometry draws are not bit-matched; their distribution and sampling seed are unchanged. This is a single-seed comparison. Full logs include losses, gradients, embedding dispersion and durations.

Frozen RGB/gray baselines use the actual 576-D pooled ImageNet features with no random projection. The same frozen partitions, image-level ranking, stable tie rule and pair lists apply to all models. No architecture decision used test results. All model and calibration hashes were locked before one final test invocation; the earlier v1 test was invalidated after duplicate-screening misses. The corrected v2 protocol was evaluated once after locking, but prior corpus exposure means it is not a pristine untouched holdout. See CORRECTION_LOG.md.

## Controlled transformations

Primary recoloring permutes RGB channels, selecting one of three nonidentity permutations using seed 17 plus query index. Secondary grayscale removes chroma. Secondary inversion maps RGB to 255−RGB, preserving boundaries while reversing contrast polarity. Inversion was added before final locking because validation on channel permutation saturated. All are deterministic derivatives of the gallery photograph; none is a new capture or real colorway. Each model uses its primary validation-recolor threshold unchanged across secondary stresses.

## Retrieval results

Ranks are images, not aggregated designs. Recall@K is the fraction of eligible queries with a positive in the first K ranks. AP averages precision at positive ranks divided by all eligible positives; mAP averages AP over eligible queries. With one positive per query here, AP is reciprocal rank.

| Model | Stress | Recall@1 | Recall@5 | mAP |
|---|---|---:|---:|---:|
| gray_gray | recolor | 1.0000 | 1.0000 | 1.0000 |
| gray_gray | grayscale | 1.0000 | 1.0000 | 1.0000 |
| gray_gray | inversion | 1.0000 | 1.0000 | 1.0000 |
| no_color_trained | recolor | 0.9778 | 1.0000 | 0.9889 |
| no_color_trained | grayscale | 0.9778 | 1.0000 | 0.9889 |
| no_color_trained | inversion | 1.0000 | 1.0000 | 1.0000 |
| rgb_rgb | recolor | 0.9556 | 1.0000 | 0.9778 |
| rgb_rgb | grayscale | 1.0000 | 1.0000 | 1.0000 |
| rgb_rgb | inversion | 1.0000 | 1.0000 | 1.0000 |
| trained_trained | recolor | 1.0000 | 1.0000 | 1.0000 |
| trained_trained | grayscale | 1.0000 | 1.0000 | 1.0000 |
| trained_trained | inversion | 1.0000 | 1.0000 | 1.0000 |

## Verification results

Threshold maximizes validation balanced accuracy; highest threshold breaks ties. Scores are cosine similarities, not probabilities. These synthetic thresholds do not establish natural same-design decisions and are not open-set gallery thresholds.

| Model | Stress | ROC-AUC | Precision | Recall | F1 | Balanced accuracy | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| gray_gray | recolor | 1.0000 | 0.7759 | 1.0000 | 0.8738 | 0.9967 | 0.792828 |
| gray_gray | grayscale | 1.0000 | 0.7377 | 1.0000 | 0.8491 | 0.9960 | 0.792828 |
| gray_gray | inversion | 1.0000 | 0.8036 | 1.0000 | 0.8911 | 0.9972 | 0.792828 |
| no_color_trained | recolor | 0.9989 | 0.5789 | 0.9778 | 0.7273 | 0.9808 | 0.487698 |
| no_color_trained | grayscale | 0.9993 | 0.6875 | 0.9778 | 0.8073 | 0.9838 | 0.487698 |
| no_color_trained | inversion | 0.9999 | 0.4891 | 1.0000 | 0.6569 | 0.9881 | 0.487698 |
| rgb_rgb | recolor | 0.9977 | 0.2744 | 1.0000 | 0.4306 | 0.9699 | 0.652521 |
| rgb_rgb | grayscale | 0.9991 | 0.4245 | 1.0000 | 0.5960 | 0.9846 | 0.652521 |
| rgb_rgb | inversion | 0.9997 | 0.2601 | 1.0000 | 0.4128 | 0.9677 | 0.652521 |
| trained_trained | recolor | 0.9999 | 0.7759 | 1.0000 | 0.8738 | 0.9967 | 0.720214 |
| trained_trained | grayscale | 0.9999 | 0.8302 | 0.9778 | 0.8980 | 0.9866 | 0.720214 |
| trained_trained | inversion | 0.9996 | 0.7857 | 0.9778 | 0.8713 | 0.9859 | 0.720214 |

## Interpretation and error counts

Channel permutation retains exact photographic geometry and many texture cues; strong results can reflect shortcuts rather than design recognition. The measured tables, not the proposed architecture, determine comparative claims. No superiority on real designs is established. Frozen grayscale is a serious baseline, but even grayscale is not invariant to all palette luminances or contrast polarity.

| Model | Recolor top-1 errors | Grayscale top-1 errors | Inversion top-1 errors |
|---|---:|---:|---:|
| gray_gray | 0 | 0 | 0 |
| no_color_trained | 1 | 1 | 0 |
| rgb_rgb | 2 | 0 | 0 |
| trained_trained | 0 | 0 | 0 |

Private score matrices and opaque-ID error lists support inspection without sharing images. No uncertainty interval was estimated; image-sharing pair dependence prevents treating 2,025 pairs as independent examples. Hard different-design/similar-palette performance is unavailable without reliable design truth. The synthetic geometric software fixtures use identical palettes but are not dataset evidence.

## Executed versus not executed

Executed: metadata/image audit, frozen group split, four-epoch real-image instance training and ablation, frozen baselines, validation calibration, locked final synthetic evaluation, checkpoint checks, gallery/inference, CPU efficiency, tests and explicit synthetic notebook execution. Not executed/supported: real identity-supervised training, natural independent-capture design metrics, real alternate-colorway metrics, CUDA/MPS runs, independent human validation, publication or form submission. See SOURCES.md and AI_DISCLOSURE.md.

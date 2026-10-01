"""Create clean, executable notebook with explicit real/synthetic modes."""
from pathlib import Path
import nbformat as nbf
nb=nbf.v4.new_notebook();cells=[]
md=lambda s:cells.append(nbf.v4.new_markdown_cell(s))
code=lambda s:cells.append(nbf.v4.new_code_cell(s))
md('''# DeepLure: color-invariant saree design recognition
This repository implements image-level identification and pair verification with a compact PyTorch embedding model. **AI tools used: Yes.** Read `AI_DISCLOSURE.md` and the execution report before presenting results.

Exact design means distinctive motif and spatial layout, including visible border/pallu. Category labels are not design labels. The executed experiment uses real public images for instance-contrastive training; its held-out recoloring benchmark is synthetic, not natural cross-colorway evidence.

This submission notebook has clean outputs. A separately executed smoke notebook is software evidence only. Real-data results from the coding-agent run are recorded in `reports/`. The first experiment was invalidated after duplicate-screening misses; the corrected experiment and prior test exposure are disclosed in `reports/CORRECTION_LOG.md`. This notebook requires the repository/ZIP contents and authorized local data; it does not download proprietary data or silently substitute fixtures.''')
code('''from pathlib import Path
import os, sys, subprocess, json, datetime
# Place this notebook inside the extracted repository's notebooks/ directory.
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "saree").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    raise FileNotFoundError("Extract the full project ZIP; run from its root or notebooks directory.")
MODE = "real"  # explicit alternatives: "real" or "synthetic_smoke"
DATA_ROOT = PROJECT_ROOT / "private/data"
OUTPUT_ROOT = PROJECT_ROOT / "private/notebook_run"
DEVICE = "auto"  # cpu, cuda, mps, auto
TRAINING_MODE = "instance"  # supervised requires independently supported design/capture annotations
ANNOTATIONS = None  # optional private JSON mapping relative paths to evidenced annotations
CONFIG = PROJECT_ROOT / "configs/cpu.json"
INSTALL_DEPENDENCIES = False  # set True only if this kernel has not been bootstrapped
if MODE not in ("real", "synthetic_smoke"):
    raise ValueError("Explicit mode required")
if INSTALL_DEPENDENCIES:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / "private/torch-cache")
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "private/matplotlib-cache")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
if OUTPUT_ROOT.exists():
    raise FileExistsError("Use a new OUTPUT_ROOT to preserve frozen splits and one-use test results.")
OUTPUT_ROOT.mkdir(parents=True)
from saree.common import read_json, write_json, device_for, seed_all
seed_all(17)
device = device_for(DEVICE)
config = read_json(CONFIG)
print({"mode": MODE, "device": str(device), "training_mode": TRAINING_MODE})''')
md('''## Audit and protocol
Data discovery and licensing are documented in `reports/AUDIT.md` and `reports/SOURCES.md`. Before using supervised mode, establish exact design and capture provenance; folder names are insufficient. Unknown information remains null.

The instance fallback groups exact/pixel duplicates, documented derivatives and likely near-duplicates before a seeded approximately 70/15/15 group split. One representative per group is retained. It does **not** establish unseen-design generalization. Vendor data is conservatively excluded in this fallback. Natural mode requires distinct capture groups on gallery/query sides of the same held-out designs.

Synthetic query images are deterministic RGB-channel permutations of held-out gallery originals. Secondary grayscale and RGB-complement (255-x) stress tests use the same frozen threshold. These copies are never represented as genuine colorways. Different instance groups are provisional negatives; hidden shared designs may remain.''')
code('''from saree.data import audit, make_splits, policy_annotations
from saree.fixtures import generate
annotations = read_json(ANNOTATIONS) if ANNOTATIONS else {}
if MODE == "synthetic_smoke":
    DATA_ROOT = OUTPUT_ROOT / "fixtures"
    generate(DATA_ROOT)
    TRAINING_MODE = "instance"
    config.update(epochs=2, batch_size=4, max_seconds=120, pretrained=True, synthetic_smoke=True)
else:
    if not DATA_ROOT.is_dir() or not any(p.suffix.lower() in (".jpg", ".jpeg", ".png") for p in DATA_ROOT.rglob("*")):
        raise FileNotFoundError("Authorized real images missing. Run scripts/acquire.py or supply DATA_ROOT. No synthetic fallback.")
    policy = policy_annotations(DATA_ROOT, PROJECT_ROOT / "configs/audit_policy.json", exclude_vendor=TRAINING_MODE=="instance")
    policy.update(annotations)
    annotations = policy
write_json(OUTPUT_ROOT / "annotations.json", annotations)
manifest = OUTPUT_ROOT / "manifest.json"
split = OUTPUT_ROOT / "split.json"
audit_summary = audit(DATA_ROOT, manifest, OUTPUT_ROOT / "annotations.json")
split_summary = make_splits(manifest, split, TRAINING_MODE, seed=config["seed"])
print({k:audit_summary[k] for k in ["images","corrupt","exact_pixel_duplicates","reliable_designs"]})
print(split_summary)''')
md('''## Model, loss and training
MobileNetV3-Small `IMAGENET1K_V1`, adaptive mean pooling, learned 576→256 projection, L2 normalization. Frozen baselines use the actual 576-dimensional pooled features, in RGB and grayscale, with no learned head.

EXIF correction, RGB conversion, aspect-fit bilinear resize and neutral padding to 224×224 preserve the full view. ImageNet mean/std normalization follows the checkpoint; replacing its center crop is deliberate. The upstream Kaggle stretch cannot be undone. Color jitter and occasional grayscale are independent for each view; geometry is limited to small translations. Palette permutation is an evaluation stress transform.

Contrastive loss excludes self-similarity. Reliable supervised mode uses P identities × K distinct captures, then two augmented views per capture. Instance mode uses two views of each original; these are not independent observations. AdamW, one head-only warm-up epoch, then backbone fine-tuning with fixed batch-normalization statistics. The highest validation mAP selects the checkpoint, earliest on ties. CPU uses float32; optional CUDA AMP is supported.''')
code('''from saree.training import train, baseline
checkpoint = Path(train(DATA_ROOT, split, OUTPUT_ROOT / "trained", config, device))
checkpoints = [checkpoint]
if MODE == "real":
    for kind in ("rgb", "gray"):
        ck = OUTPUT_ROOT / kind / "baseline.pt"
        baseline(kind, split, ck)
        checkpoints.append(ck)
print(read_json(OUTPUT_ROOT / "trained/training_summary.json"))''')
md('''## Validation calibration, lock, and final evaluation
Recall@K means the fraction of eligible queries with at least one relevant gallery image in the first K ranks. AP averages precision at relevant ranks over all gallery positives; mAP averages AP across eligible queries. Queries without positives are counted separately. Rankings are image-level, with stable manifest-order ties.

Verification uses the frozen pair lists and cosine scores. The threshold maximizes validation balanced accuracy, breaking ties toward the highest threshold. All model choices and thresholds are locked before a single final test invocation. ROC-AUC, precision, recall, F1 and balanced accuracy are reported with class counts. Synthetic thresholds are not proven natural same-design thresholds. Pairwise thresholds are not open-set gallery rejection thresholds.

No unsupported low-FPR or independent-pair uncertainty claim is made. Pair observations share images. Natural cross-colorway and hard different-design/similar-palette results remain unavailable unless supported by actual annotations.''')
code('''from saree.evaluation import calibrate, lock_models, evaluate
calibrations = []
for ck in checkpoints:
    cal = ck.parent / "calibration.json"
    calibrate(ck, split, DATA_ROOT, cal, device)
    calibrations.append(cal)
lock = OUTPUT_ROOT / "experiment_lock.json"
lock_models(split, checkpoints, calibrations, lock)
results = evaluate(lock, split, DATA_ROOT, OUTPUT_ROOT / "evaluation", device)
for name, value in results.items():
    key = "recolor" if TRAINING_MODE == "instance" else "natural"
    print(name, value[key]["retrieval"], {k:v for k,v in value[key]["verification"].items() if k != "roc"})''')
md('''## Identification, pair verification and efficiency
Gallery embeddings are cached once. Cache metadata binds the checkpoint, preprocessing, frozen split and embedding-file hashes. Exact cosine matrix search returns opaque IDs and scores, not probabilities. Uploads in the optional local app stay local.

The benchmark measures encoder, preprocessing (excluding disk decode), and exact search separately with warm-up/repeats and device synchronization. Parameters, embedding storage and checkpoint size are measured. FLOPs are unavailable because no FLOP-counting tool was run.''')
code('''from saree.retrieval import build_gallery, retrieve, verify
from saree.benchmark import benchmark
cache_dir = OUTPUT_ROOT / "gallery"
build_gallery(checkpoint, split, DATA_ROOT, cache_dir, device=device)
spec = read_json(split)
row = next(r for r in spec["rows"] if r["image_id"] == spec["protocols"]["test"]["gallery"][0])
example = DATA_ROOT / row["relative_path"]
# API sanity check on one gallery image; not an additional evaluation metric.
print(retrieve(checkpoint, cache_dir, split, example, device=device))
print(verify(checkpoint, calibrations[0], example, example, device=device))
print(benchmark(checkpoint, OUTPUT_ROOT / "efficiency.json", device=device))''')
md('''## Limits, source disclosures and follow-up
Study `reports/EVALUATION.md`, `reports/EFFICIENCY.md`, `INTERVIEW_GUIDE.md`, and `DECISIONS.md`. Similar color can still dominate; grayscale can erase isoluminant boundaries. A 224-pixel global representation loses tiny motifs, and backgrounds/cloth folds can shortcut instance matching. Genuine annotated alternate colorways and independent captures are the next evidence needed.

Sources: [assignment dataset on Kaggle](https://www.kaggle.com/datasets/div456/indian-saree-patterns), its bundled Roboflow README, and [torchvision MobileNetV3-Small documentation](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.mobilenet_v3_small.html). The checkpoint is `mobilenet_v3_small-047dcff4.pth` from download.pytorch.org. No external vision API processed corpus images. The vendor link is intentionally omitted from shareable code.

**AI tools used: Yes.** All visual inspection by the coding agent is AI-assisted, not human-verified. This clean notebook is an executable deliverable, not itself proof of execution. Do not embed private images in notebook outputs or upload vendor data to hosted runtimes without permission. Delete project vendor copies after the exercise. The user must verify reviewer access and submit personally by **2 October 2026, 11:59 AM IST**.''')
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
nbf.write(nb,'notebooks/DeepLure_Submission.ipynb')
print('Wrote clean submission notebook')

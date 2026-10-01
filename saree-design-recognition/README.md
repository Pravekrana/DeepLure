# Color-invariant saree design recognition
PyTorch image retrieval and pair verification for the DeepLure assignment. Exact motif/layout identity is the target, not category classification.

**Execution status:** real public images were acquired and audited; the implemented instance-contrastive fallback and two pretrained baselines are evaluated on controlled synthetic recoloring. Natural distinct-capture/cross-colorway performance is unavailable because reliable design, capture, and colorway labels were not established. Read the final `reports/EVALUATION.md` for executed results. **AI tools used: Yes.** This is not a claim of unaided authorship or submission readiness.

## Environment
Tested on Python 3.12, CPU PyTorch 2.9.0 and torchvision 0.24.0. Tested direct dependencies are in `requirements.txt`; the installed Linux CPU dependency closure is pinned in `requirements-cpu.lock.txt`, with hardware/software details in `reports/environment.json`. For a fresh CPU environment, install that lock with `--extra-index-url https://download.pytorch.org/whl/cpu`. Keep an existing working accelerator stack when using other devices. CUDA and compatible MPS devices are supported by the code but were not tested here.

```bash
cd saree-design-recognition
python3 scripts/init_private.py
python3 -m venv --system-site-packages .venv
# Preserve a working compatible PyTorch installation. If none is installed, CPU setup:
# .venv/bin/python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
. scripts/env.sh
.venv/bin/python -m pytest -q
```

In the agent's workspace, `.venv` points to `/tmp/deeplure-runtime` because installing large environments on the Windows-mounted filesystem was very slow. It is a local, temporary environment, not an archive dependency; recreate it after a reboot. No system-wide packages or settings were modified. `scripts/env.sh` directs model and plotting caches to private project storage.

## Data and privacy
Initialize the empty private workspace using `scripts/init_private.py` before acquisition; it refuses to claim existing unmarked files. For the mixed-source protocol, DATA_ROOT contains `kaggle/` and optionally `vendor/` subdirectories, as created by the acquisition script. On Kaggle, download/stage the permitted public source under the same structure and set DATA_ROOT/OUTPUT_ROOT accordingly.

Only acquire data you are authorized to use. Keep images and derived artifacts local under `private/`. For the supplied public source:

```bash
.venv/bin/python scripts/acquire.py --source kaggle --data-root private/data
```

For vendor data, `scripts/acquire.py --source vendor --vendor-url "$AUTHORIZED_VENDOR_URL" --data-root private/data` accepts the privately supplied assignment link; do not put that link into public files. The corpus must be deleted after the exercise, using the scoped procedure in `DATA_RETENTION.md`. No vendor uploads, public hosting or publication is performed. Kaggle's API and bundled README disagree about the public dataset license (CC0 vs MIT); images and weights are therefore excluded from the ZIP.

The optional private annotation JSON maps relative image paths to supported `design_id`, `colorway_id`, `capture_group`, `annotation_provenance`, `annotation_confidence`, and `excluded_reason`. Use opaque IDs, mark unknown as null, and do not label categories as designs. High-confidence design annotations require explicit evidence. See `reports/AUDIT.md`.

## Reproduce the executed protocol
Use a new output directory for a fresh run; frozen split files and final test outputs cannot be overwritten. The notebook creates vendor exclusions in instance mode automatically. For the agent's existing audited data, the exact pipeline commands are:

```bash
. scripts/env.sh
# These audit/split steps were already executed in the local workspace.
.venv/bin/python scripts/prepare_annotations.py --data-root private/data --out private/annotations_v2.json
.venv/bin/python -m saree.cli audit --data-root private/data --annotations private/annotations_v2.json --out private/manifest_v2.json
.venv/bin/python -m saree.cli split --manifest private/manifest_v2.json --out private/split_v2.json --mode instance --seed 17
.venv/bin/python -m saree.cli --device cpu train --data-root private/data --split private/split_v2.json --out private/run_v2/trained --config configs/cpu.json
.venv/bin/python -m saree.cli --device cpu train --data-root private/data --split private/split_v2.json --out private/run_v2/no_color --config configs/no_color.json
.venv/bin/python -m saree.cli baseline --kind rgb --split private/split_v2.json --out private/run_v2/rgb/baseline.pt
.venv/bin/python -m saree.cli baseline --kind gray --split private/split_v2.json --out private/run_v2/gray/baseline.pt
```

`private/annotations_v2.json` is intentionally not shared; `scripts/prepare_annotations.py` regenerates the disclosed exclusions from `configs/audit_policy.json`. In a fresh checkout, use `notebooks/DeepLure_Submission.ipynb` to generate the conservative exclusion policy, or supply independently supported annotations. The notebook is self-contained given the extracted repository and authorized data; its configuration cell exposes paths, mode, device and configuration. Default mode is real; missing assets raise an error. The separate executed smoke notebook uses generated geometric fixtures and is not a real-data result.

Calibrate each model on validation, then lock all choices before final testing:

```bash
.venv/bin/python scripts/finalize_experiment.py --run-root private/run_v2 --data-root private/data --split private/split_v2.json
```

This script is one-use for a run: it saves thresholds, a lock, final metrics/curves, and a compatible gallery. Test labels never select a threshold or checkpoint. Natural supervised mode is available with `split --mode supervised` once trustworthy identities/captures support it. The P×K sampler requires at least P identities with K independent capture groups; it rejects invalid coverage. Do not present that path as naturally evaluated by the current experiment.

## Inference and efficiency
```bash
.venv/bin/python -m saree.cli gallery --checkpoint private/run_v2/trained/best.pt --split private/split_v2.json --data-root private/data --out private/run_v2/gallery
.venv/bin/python scripts/demo_inference.py --run-root private/run_v2 --split private/split_v2.json --data-root private/data
.venv/bin/python -m saree.cli benchmark --checkpoint private/run_v2/trained/best.pt --out private/run_v2/efficiency.json
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

The demo command selects existing private gallery paths and exercises both `retrieve` and `verify` CLI commands; it is an API sanity check, not evaluation. For new images, `python -m saree.cli retrieve --help` and `verify --help` list explicit image arguments.

Retrieval returns image-level opaque IDs and cosine similarities. There is no gallery open-set rejection rule. Pair decisions use the saved validation threshold and warn when calibration is synthetic. Cosine scores are not probabilities. The app shares the same model, preprocessing, gallery validation and pair-verification code; no fake predictions appear for missing assets.

## Software-only integration and packaging
```bash
.venv/bin/python scripts/smoke.py --out private/new_smoke
.venv/bin/python scripts/execute_smoke_notebook.py
.venv/bin/python scripts/package.py
.venv/bin/python scripts/cleanup.py  # dry run only
```

The notebook smoke uses the downloaded ImageNet checkpoint on generated geometric fixtures. Use a fresh output directory when repeating it. Packaging reads `SHAREABLE_FILES.txt`, rejects missing entries and image-bearing notebook outputs, scans private-access patterns, and creates `dist/DeepLure_Submission.zip`. Private manifests, images, weights, caches and acquisition metadata are never allowlisted.

Study `INTERVIEW_GUIDE.md`, `DECISIONS.md`, `AI_DISCLOSURE.md`, `DATA_RETENTION.md` and `SUBMISSION_CHECKLIST.md`. The user must arrange and verify an authorized reviewer-accessible link and personally submit the form by **2 October 2026 at 11:59 AM IST (06:29 UTC)**. A local ZIP path is not a submission link.

# Progress
Updated 2026-10-01 14:04 UTC. Deadline: 2026-10-02 06:29 UTC (11:59 AM IST).

**Current result: corrected v2 implementation, experiments and package complete locally. Natural design/colorway performance remains unproven; no hosted link or form submission exists.**

- Audited 1,468 Kaggle + 165 vendor images. Excluded unresolved vendor capture families and four known furniture-render source variants. Corrected split: 211 train / 46 validation / 45 test public-source representatives.
- Ran two four-epoch instance-contrastive experiments and frozen RGB/gray baselines. Validation selected epoch 2 for color augmentation and epoch 1 for no-color. Trained primary threshold: 0.72021425.
- Corrected synthetic recoloring test: trained and gray both Recall@1/mAP 1.0 and F1 0.87379; RGB Recall@1 0.95556 and F1 0.43062. These compare derivatives of the same photographs, not genuine colorways.
- V1 was invalidated after a missed near-duplicate crossed validation/test. Added pixel screening, rebuilt splits and retrained unchanged configurations. Previous corpus/test exposure remains disclosed in reports/CORRECTION_LOG.md; no pristine-holdout claim.
- Seventeen tests passed locally and from a freshly extracted ZIP. The clean submission notebook's five code cells executed in explicit synthetic mode in the separate smoke notebook. Real-data execution used the repository pipeline. V2 app, CLI inference, missing-asset behavior, dependency consistency and source-byte integrity checks passed.
- CPU encoder: 23.216 ms median, batch 1 at 224×224; 1,074,720 parameters; 256 float32 embedding values. Full details in reports/EFFICIENCY.md.
- Shareable ZIP uses 76 explicit allowlisted files; images, weights, private manifests/access metadata and caches are excluded. Approach length: 498 characters. AI tools used: Yes.

Private assets are under private/run_v2; the superseded run is retained under private/run. Model/data paths and commands are in README.md. The project environment points to /tmp/deeplure-runtime and must be recreated after its removal/reboot.

Remaining: reliable natural design/capture/colorway annotations, recruitment-email review, user study/review, and an authorized reviewer-accessible link with access verified before personal form submission. Preserve working data until the exercise concludes, then use the scoped cleanup procedure. No system settings, paid compute, public dataset upload or personal attestation was performed.

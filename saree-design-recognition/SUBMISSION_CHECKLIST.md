# Submission checklist
**Status: implemented, locally executed; final packaging checks are recorded in dist/package_audit.json and dist/extracted_verification.json; NOT submission-ready as a proven natural colorway system.** No reviewer-accessible URL has been created or verified, and no form was submitted.

Deadline: **2 October 2026 at 11:59 AM IST = 06:29 UTC**, not 11:59 PM. Runtime clock was checked at start and during execution. There is no assumed extension.

- [x] Read local ps_v3.pdf and implement identification plus verification in PyTorch.
- [x] Acquire/audit 1,468 public-source and 165 vendor images without public uploads.
- [x] Investigate metadata and perform AI-assisted visual inspection; preserve unknown labels and private evidence.
- [x] Freeze grouped train/validation/test manifests before augmentation/model fitting.
- [x] Execute real-image instance-contrastive training and no-color ablation, four epochs each; evaluate frozen RGB/gray baselines.
- [x] Select epochs and thresholds only on validation; lock models before each recorded final synthetic test pass; v1 invalidated after duplicate audit correction, v2 rerun disclosed.
- [x] Report actual retrieval/verification metrics, class counts, negative precision limitations and CPU efficiency.
- [x] Execute software tests, train-to-retrieval smoke, clean notebook's explicit synthetic mode, and local app checks.
- [x] Prepare clean submission notebook, separately executed smoke notebook, <=500-character approach, disclosure, study guide and retention procedure.
- [x] Package code and aggregate results by explicit allowlist; exclude all dataset images, private manifests/access metadata, caches, and checkpoints.
- [ ] Establish independently supported exact-design, capture and genuine-colorway annotations; natural recognition metrics remain unavailable. The existing synthetic threshold does not validate natural pair decisions.
- [ ] Supply/review the recruitment email, which was not found in the workspace or provided in the conversation.
- [ ] Human review and personal understanding: user must study code/results, reproduce chosen checks, and decide whether this qualified submission meets the interview requirements.
- [ ] Create an authorized reviewer-accessible repository or notebook link. Suggested route: upload only the allowlisted ZIP contents to a repository under the user's control, or place the clean notebook together with its required package/config files in an approved notebook/repository environment. Never upload vendor data or privately derived weights without permission.
- [ ] Arrange a private authorized artifact/data route if reviewers need the exact local weights. Alternatively supply reproducible acquisition/training commands for authorized sources. Public code alone does not imply publication rights for data/checkpoints.
- [ ] Verify link access using a logged-out/private browser or a reviewer-equivalent account: open the notebook and required files, confirm no credentials are needed beyond explicitly granted reviewer access, and ensure dependencies/data instructions are visible. A local file path is not a submission link.
- [ ] Submit personally at https://forms.gle/2tff1iKAvxoKh2xaA before the deadline. The agent must not submit or make personal attestations.

**AI tools used: Yes. Form answer: Yes.** See AI_DISCLOSURE.md; no unaided authorship or human/independent validation is claimed.

After the exercise and review conclude, execute the dry-run cleanup first, then the explicitly scoped deletion in DATA_RETENTION.md. Do not delete working data prematurely or remove original user files.

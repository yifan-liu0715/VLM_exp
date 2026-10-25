# Nina workspace snapshot

Source: `/nvmedata/workspace2/users/nina` on the user's research server. Snapshot date: October 25, 2026.

## Current preservation status

- Source inventory: 233,053 regular files, 479,313,035,408 bytes.
- This Git snapshot contains 32,009 source/code/configuration/log/manifest/text files under `workspace/`, preserving their relative paths. Exported content hashes are in `metadata/text_export_manifest.sanitized.jsonl`.
- Video, weights, installed environments, caches and other binary payloads are **not yet uploaded**. Their 194,807 source files (439,253,291,370 bytes) are represented in a 410-job Release transfer plan, not by fake completed backup records.
- `dataset_h3_public_v1` contributes manifests, screening/review results and reports only. Public dataset payloads are excluded.
- A credential file and transient sockets/pipes are excluded. Eight sensitive values found in historical code/logs were redacted in 14 exported text files; source files on the server were not edited. The report contains no credential values.
- One nested-Git-history archive is held for separate safety review. Installed environments and binary archives also require content safety checks before publication; a prepared transfer plan alone does not certify them.

The repository is a preservation snapshot. Existing candidate/accepted/review-needed states are preserved. Uploading files does not mean the dataset has passed current semantic, privacy, keypoint, or training readiness checks, and does not mean model training has completed.

## Layout and classification

- `workspace/batch_runs/`: batch processing code, logs, proposals, review metadata and run reports. Raw/proxy/clip videos remain in the Release plan until uploaded.
- `workspace/h3_training_v1/`: H3 training and testing code/configurations/run metadata.
- `workspace/prj/`: prior training/test projects, scripts, experiment reports and configuration files. Model weights, binaries and environments are planned as Release assets.
- `workspace/dataset_h3_capture*/`: capture dataset manifests, keypoint JSON and screening/review records. Video payloads are planned separately.
- `workspace/dataset_h3_public_v1/`: permitted public-dataset manifests and screening results.
- `metadata/`: full source inventory, explicit exclusions, directory/symlink metadata, source/export hashes, redaction report and Release transfer plan.
- `tools/`: transfer planning, resumable Release publishing, reconstruction and verification helpers. One-shot preparation scripts are retained as provenance.

`metadata/inventory.jsonl` lists source regular files; `metadata/directories.jsonl` and `metadata/symlinks.jsonl` preserve non-file structure. `metadata/exclusions.jsonl` records every intentional exclusion. The inventory does not imply that an asset has been uploaded: only an `uploaded_verified` Release receipt does.

## Continuing the binary upload

The Git push uses the user's existing SSH authentication. The Releases API additionally needs a GitHub CLI login as `yifan-liu0715`. No login token is stored in this repository or in manifests. The publisher does not change billing settings or upload data to Git LFS.

On the user's Windows machine, log into GitHub CLI. The agent must finish the pending payload content scan before starting the publisher; it refuses to upload while that scan is pending:

```powershell
gh auth login --hostname github.com --git-protocol ssh --web --skip-ssh-key
python 'G:\h3_project_backup_20261025\github_export_20261025\tools\publish_release.py' --workers 2
```

The publisher uses `h3-server` and the dedicated server export directory `/nvmedata/workspace2/users/nina_export_20261025`. It creates temporary assets there, verifies original file size/mtime before and after reading, uploads each asset into this repository's `nina-workspace-20261025` Release, verifies GitHub's reported size and SHA-256 (or performs a download verification), and then removes only the generated temporary copy. Originals remain untouched. API credentials travel in the worker's standard input and are never written to disk or command arguments.

Safety-held payloads must be reviewed before being enabled. The initial scripts intentionally stop on a source mutation, conflicting asset, checksum mismatch or failed upload. Logs/status and verified receipts are saved under the export's `metadata/` directory. Upload receipts should be committed after each completed batch; a final completion claim requires reconciling every inventory row with the Git tree, a verified Release receipt or an explicit exclusion.

## Restoring

Clone the Git repository for code/text first. Once verified Release receipts exist, use:

```powershell
python tools/restore_release.py --destination 'G:\restored_nina' --download
```

The restorer verifies asset SHA-256, reconstructs split files from their original byte offsets, rejects path traversal and unsafe archive entries, and refuses to overwrite different existing files. It needs Python 3.10+ and `zstd`. The prepared transfer/restore path has passed eight local-on-server safety/roundtrip checks; the real 439 GB payload has not yet been uploaded or restored.

Environment bytes are preserved for reference, but Conda prefixes and external symlink targets may not be portable. External targets are recorded only and are never silently copied from outside the requested workspace.
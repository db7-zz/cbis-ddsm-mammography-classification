# Phase 4 formal artifact freeze record

Freeze date: 2026-07-21
Formal evaluation date: 2026-07-17
Status: verified read-only snapshot

## Version boundary

- Git merge commit: `d6530e1f7a53fc131e3a15c6b39f7258599a49de`
- Git tag: `phase-4-final-evaluation`
- Pull request: `#1`, merged into `main` on 2026-07-20
- Formal test inference passes: `1`

The tag is the immutable source-code boundary. The artifact snapshot is the
runtime evidence boundary. Later experiments must not overwrite, resume into,
or retrospectively tune this baseline.

## Snapshot location and scope

Local snapshot:

```text
backups/phase4_final_frozen_20260721/
```

The snapshot contains 35 files (417,927,198 bytes / 398.57 MiB):

- unified train/validation/test manifests and data-integrity reports;
- formal training configuration, epoch metrics and TensorBoard event;
- `best.pt` and `last.pt` checkpoints;
- best validation predictions;
- validation-frozen threshold and threshold-selection audit;
- official-test predictions, metrics, outcome and figures;
- tracked model/train/evaluation configurations;
- phase 1-4 specifications and reports;
- project dependency specification and README.

The approximately 160 GB raw DICOM collection is intentionally excluded. It is
recoverable from the official source and is validated through the frozen
manifests, UID mapping and integrity records.

## Freeze procedure

The repeatable command is:

```powershell
python scripts/freeze_phase4_artifacts.py
```

The script implements the following controls:

1. Uses an explicit allowlist of formal artifacts. It does not recursively copy
   arbitrary project files.
2. Refuses to run if the destination already exists, preventing an earlier
   freeze from being overwritten.
3. Computes SHA-256 for every source file before copying.
4. Copies each file while preserving timestamps and verifies destination size
   and SHA-256 immediately.
5. Writes `freeze_manifest.json`, `freeze_manifest.sha256` and a read-only
   warning file.
6. Marks every copied file read-only.
7. Marks 22 formal runtime source files below `artifacts/` read-only. Tracked
   configs and documents remain writable because their immutable version is the
   Git tag.
8. Re-reads every copied file after freezing and repeats the hash and permission
   checks.

## Independent verification result

| Check | Result |
|---|---:|
| Manifest file count | 35 |
| Backup files marked read-only | 35 / 35 |
| Formal source files marked read-only | 22 / 22 |
| Source/backup SHA-256 mismatches | 0 |
| Manifest checksum match | Yes |

Freeze manifest SHA-256:

```text
61176600394d4abdfc62aad075c9c2aa1567f9f03e654ff6117d89e6a867de30
```

The detailed per-file hashes are stored in:

```text
backups/phase4_final_frozen_20260721/freeze_manifest.json
```

## Recovery and integrity policy

Before restoring any artifact, first verify `freeze_manifest.sha256`, then
verify each restored file against the per-file SHA-256 list. Restore into a new
directory; do not replace the frozen source in place.

The current snapshot is on the same D: volume as the working project. It
protects against accidental overwrite and output-directory reuse, but it is not
a disaster-recovery copy. For true recovery, the complete frozen directory must
also be copied unchanged to another physical disk or managed object storage.

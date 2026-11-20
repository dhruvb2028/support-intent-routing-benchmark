# Data assets

**Bitext source and derived corpus: CDLA-Sharing-1.0.** Original provider: Bitext Innovations. Source revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443`. This directory contains modified derivatives: canonical duplicates removed, request/intent columns retained and renamed, example/group identifiers added, and deterministic partitions assigned. These data changes are documented in `../docs/data-report.md`; retain this attribution and the original data license when redistributing. The official license copy is `LICENSE-CDLA-Sharing-1.0.html`; the unmodified source card is `raw/README.md`. Code licensing does not replace the source data license.

- `processed/train.csv`, `validation.csv`, `test.csv`: exactly example_id, text, label, group_id. Use only text as model input. Frozen v2 corpus counts: 16,681 / 3,575 / 3,570.
- `manifest.json`: pinned source hashes, grouping/audit evidence, split hashes and feature contract.
- `processed/challenge_development.csv`: 60 newly AI-authored requests for threshold development.
- `processed/challenge_test.csv`: 300 newly AI-authored final diagnostic requests; do not tune on their outcomes. Includes 180 supported, 60 ambiguous, 30 near-domain unsupported, 30 distant unsupported requests.
- `challenge_manifest.json`, `ontology.json`: challenge provenance, scope and actions. Metadata and gold labels are never features.
- `review/`: independent AI label review records, not human annotation.
- `audit/source_row_map.csv`: retained example IDs mapped to zero-based original row indices.

Regenerate from the project root: `python scripts/prepare_data.py`. Regenerate authored challenge assets with `python data/challenge_authoring.py`; do not overwrite frozen final data after observing model outcomes. Hash checks refuse modified raw data.

The original corpus and authored challenge are synthetic. Neither represents production customer traffic. See `../docs/data-report.md` for evidence and limitations.

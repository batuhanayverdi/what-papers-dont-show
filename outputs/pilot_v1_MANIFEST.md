# Pilot v1: audited artifact (superseded by Pilot v2)

**Status:** audited and kept unchanged for reference and for the v1 vs v2 comparison. Do not use it for substantive claims. It was superseded by Pilot v2 (`scripts/run_pipeline_v2.py`).

**Audit:** [`pilot_v1_audit.md`](pilot_v1_audit.md)

## Known defects

1. **Substeps never reached the model.** The substeps join used `task + level + domain + field + subfield`. `substeps.csv.gz` leaves `domain` empty for subfield tasks, so 0 of 4,938 tasks matched and every prompt said "(No substeps available.)".
2. **Category was shown to the model** ("Task category:" in the prompt). Category explains 67% of the variance in v1 observability.
3. The pipeline computed `researcher_paper_gap` (`pct_researchers/100 − prevalence`), but it is not a valid measure of hidden work because the two measures have different denominators (see the audit, section 6).

## Files (SHA-256 at the time of audit; the v2 pipeline only reads them)

| File | SHA-256 |
|---|---|
| `scripts/run_pipeline.py` | `9e694f0bbb12ea78fd13a1e01dfa611b5590830bcda3e4849af8536d19e59dec` |
| `data/processed/pilot_sample.csv` | `b1564c7da6751a96303b711fe56b3082b70949918532bbc1b95c1ccac58e8a28` |
| `data/processed/openai_observability.csv` | `ba365a017aa268dfb5faa44cc26a2af89a9f545e7704aaa221c9ecc22c968361` |
| `data/processed/analysis_ready.csv` | `d08c0fd0ce9edc35015f4ad5a91dafe2a8952e10c12d410a125a2d7383d23f37` |
| `outputs/pipeline_summary.txt` | `6539d47c505ed36e8d16a3746dcf855bacb9338a5dd789f72b442506c0f2138f` |
| `outputs/pilot_v1_audit.md` | `46630ea48d2ad202fb32259b0c043e57dc9be0e4da6d2035e035563fda2bafd8` |

Check with `sha256sum <file>`. The v1 files keep their original paths because `scripts/run_pipeline.py` refers to them and Pilot v2 reuses `pilot_sample.csv` as its frozen sample.

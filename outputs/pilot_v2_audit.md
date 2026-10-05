# Pilot v2 audit

> **Status: prepared and validated, NOT labelled.**
> Every stage that does not need the OpenAI API has been run and checked: universe, fixed substeps join, frozen sample, prompt construction, and leakage checks. The labelling stage could not run here: the session has no `OPENAI_API_KEY` and its network policy blocks `api.openai.com`.
> Sections 5–8 are therefore **pending**. No observability results exist yet, and none are shown or estimated below. The analysis code was smoke-tested on synthetic labels in a throwaway copy, and no synthetic data is in the repository.
>
> To complete the pilot:
> ```
> export OPENAI_API_KEY=...                                    # never commit it
> python scripts/run_pipeline_v2.py label                      # 50 calls → openai_observability_v2.csv
> python scripts/run_pipeline_v2.py label --variant no_substeps   # 50 calls, recommended (see §6)
> python scripts/run_pipeline_v2.py label --variant replicate     # 50 calls, recommended (see §6)
> python scripts/run_pipeline_v2.py analyze                    # analysis_ready_v2.csv, comparison, summary
> ```

## 1. What was wrong with Pilot v1

Full audit: [`pilot_v1_audit.md`](pilot_v1_audit.md). Preserved files and hashes: [`pilot_v1_MANIFEST.md`](pilot_v1_MANIFEST.md).

- **Substeps join failed silently.** The key `task + level + domain + field + subfield` matched 0 of 4,938 tasks, so all 50 prompts said "(No substeps available.)". The model rated a one-sentence task statement.
- **Category leaked into the prompt.** "Task category:" was shown, and category explained 67% of the variance in v1 observability. The score was largely a category proxy.
- **The gap was the wrong outcome.** `pct_researchers/100 − prevalence` mixes a per-researcher "ever does it" estimate with a per-paper share. Its variance is dominated by prevalence (r = −0.81 across all 4,938 tasks), and it is positive for 99.9% of tasks. This follows from two features of SciNet itself:
  - subfield tasks were generated at a 70%+ researcher-coverage threshold;
  - the rating prompt anchors relevance toward 90–100.

## 2. How the substeps join was fixed

I inspected the raw file:

- In `substeps.csv.gz`, subfield-level rows have `field` and `subfield` filled 100% of the time and `domain` filled **0%** of the time. This matches SciNet's data README: "`domain` is filled only for domain-level tasks".
- `(task, field, subfield)` is unique among subfield tasks. `(task, subfield)` alone is not, because two subfield names exist in two fields.

**Fix:** subfield tasks now join on `task + level + field + subfield`. The join fails loudly unless:

- the key is unique on both sides;
- substep IDs are unique within each task;
- at least 99.5% of comparable tasks match;
- every sampled task has substeps.

All substeps are included, in `S1, S2, …` order. The v1 cap of 15 is removed; the sample's maximum is 15 and the universe's is 21.

**Validation** (`outputs/pilot_v2_validation.json`):

| Check | Value |
|---|---|
| Comparable tasks | 4,938 |
| Tasks with matched substeps | 4,938 |
| % with matched substeps | 100.0% |
| Tasks with zero substeps | 0 |
| Matches under the v1 key (for comparison) | 0 |
| Sampled tasks with substeps | 50 / 50 (8–15 substeps each, median 11) |

**The guard fires as intended.** Re-running the v2 build with the v1 key raises `Substeps join failed: only 0/4938 tasks matched`.

## 3. Category leakage removed

- **Allowlist.** `build_prompt` accepts exactly `domain, field, subfield, task, substeps_text` and raises on any other field. A test passes `category` and confirms it is rejected.
- **No category label.** The template has no category label. The checks reject a line like `category:` / `Task category:`, and reject any occurrence of the record's category value that does not come from SciNet's own task or substep text.
- **Category stays in the data.** It remains in `pilot_v2_sample.csv` for analysis.
- **Independent scan of all 100 saved prompts.** None contains "Task category", a `category:` label, the v1 category line, or "(No substeps available.)". Every main prompt contains all of its substeps (`S1:` … `Sn:`).

## 4. Outcome leakage removed

The checks run before any API call, again immediately before each call, and they also run against deliberately injected leaks:

1. **Template check.** The instruction and user template contain none of: `pct_researchers`, `researcher_paper_gap`, `researcher_share`, `prevalence`, `importance`, `frequency`, `classification`, `category`. The v1 instruction said "Do not estimate task prevalence"; v2 avoids those words entirely.
2. **Identifier check.** `pct_researchers`, `researcher_paper_gap` and `researcher_share` appear nowhere in any prompt.
3. **No labelled fields.** No banned name appears as a field label (`name:` or `name =`).
4. **Provenance check.** Any occurrence of a column-name word in a prompt must come from SciNet content. The count in the prompt must equal the count in the inserted SciNet text.
5. **Self-test.** Five injected leaks must all be caught, or the run aborts: a category line, `prevalence: …`, `pct_researchers = …`, a gap sentence, and a "frequency" sentence. All five were caught.
6. **Integrity.** Each prompt is saved with its SHA-256 in `data/processed/pilot_v2_prompts.jsonl`, together with the prompt version `observability_v2` and the model. At labelling time the prompt is rebuilt, re-hashed, and must match.

**Deviation from the literal instruction, stated openly:** a plain "string not in prompt" assertion for all seven words fails for **11 of 50 tasks**. In each case SciNet's own task or substep text uses the word as ordinary domain vocabulary:

- "frequency response" in control theory;
- "behavioral frequency items" in surveys;
- "prevalence estimates" in a zoonotic-surveillance task;
- "IRB review category" in ethics;
- "perceived issue importance" in political communication;
- "documenting the reasoning for each classification" in IP law.

These carry no SciNet rating. Removing them would alter the task description. So the literal check is applied strictly to our own text and to the identifiers, and the occurrences in SciNet text are allowed only if checks 3–4 pass. They are logged per task in the validation file. The other **39 of 50 prompts pass the literal check outright**.

**Not shown to the model:** category, pct_researchers, prevalence, importance, frequency, classification, researcher_paper_gap.

**Shown to the model:** domain, field, subfield, task, substeps.

**Prompt changes besides the inputs:**

- explicit / inferable / mostly-hidden definitions;
- an instruction to judge detectability only, not how common, how often, or how valuable a task is;
- an instruction to use substeps as evidence of what the task entails;
- a new `evidence_substeps` output field, so the rationale can be traced to specific substeps.

The 0–100 anchors are worded exactly as in v1, and the model (`gpt-5.6-sol`) and call shape are unchanged, so differences stay attributable to the inputs and instructions. The raw model output is now stored for each call.

## 5. Distribution of v2 observability scores

**Pending (labelling not run).** `analyze` will report: n, mean, median, sd, IQR, min–max, share ≥ 90 and < 50, visibility-class and location counts, the share of tasks citing evidence substeps, and the score–confidence rank correlation.

## 6. V1 vs V2 comparison

**Pending.** Per-task file: `outputs/pilot_v2_v1_comparison.csv`. The summary will report:

- mean, median and sd for v1 and v2;
- Pearson and Spearman v1~v2 correlations with bootstrap CIs;
- mean change and mean absolute change, and the count of |Δ| ≥ 15;
- a score-bin table and a visibility-class crosstab;
- mean change by category (category was shown in v1 and hidden in v2);
- the 10 largest changes with both rationales.

**Design caveat.** v2 changes two things at once: substeps are added and category is removed. A main-only run cannot attribute a change to either. Nor can it separate real change from run-to-run noise, because v1 was a single unrepeated run. Two optional runs of 50 calls each, on the same 50 tasks, resolve this:

- `--variant no_substeps`: the identical v2 prompt with substeps withheld. Then:
  - `v2 − v2_no_substeps` is the **substeps effect**;
  - `v2_no_substeps − v1` is the effect of **category removal plus the new wording**.
- `--variant replicate`: a second main run. `|v2 − v2_replicate|` is the **noise floor**. A change smaller than the noise floor is not a real change.

I recommend running both before interpreting section 6.

## 7. Exploratory correlations

**Pending.** `analyze` computes Pearson and Spearman correlations, with bootstrap 95% CIs and no p-values, for:

- observability ~ prevalence, pct_researchers, frequency, importance;
- prevalence ~ frequency, importance.

Within-category correlations are computed only for categories with n ≥ 8: Data Gathering (9) and Data Analysis (8). The other eight categories have n ≤ 6 and are listed as not computed. Even n = 8–9 gives CIs too wide for anything beyond description. No significance claims will be made at n = 50. `researcher_paper_gap` is not computed or carried into any v2 file.

## 8. Examples where substeps changed the score

**Pending.**

- If the `no_substeps` variant is run, the analysis lists tasks with |v2 − v2_no_substeps| ≥ 15 as **attributable to substeps**, quoting the cited evidence substeps.
- Otherwise it lists only **candidates**: tasks with |v2 − v1| ≥ 15 that cite evidence substeps, explicitly labelled as not attributable.

## 9. Recommended design for the main sample

1. **Finish this pilot first.** Run main, no_substeps and replicate (150 calls). Proceed only if:
   - test–retest agreement is acceptable (for example Spearman ≥ 0.8 between main and replicate);
   - observability is not stuck at the ceiling, as it was in v1 (48% of scores ≥ 90).

   If either condition fails, revise the prompt before scaling.
2. **Population and sampling.** Use all 4,938 comparable subfield tasks, sampled with **probability proportional to category share**, or stratified with recorded inclusion weights. Do not use v1's equal allocation, which sampled 2 single-task categories with certainty.
   - Because subfield explains more prevalence variance than category (23% vs 3%), sample within subfields: for example 2–3 tasks from each of ~80–100 randomly drawn subfields. Within-subfield contrasts then become possible.
   - n ≈ 200–300 gives 80% power for r ≈ 0.2.
3. **Outcome.** No raw gap. Use within-subfield comparisons of `prevalence` ranks against `pct_researchers` and `frequency` ranks, and treat observability as one candidate explanation of the residual discordance.
4. **Measurement quality.**
   - Double-label a random 20% subset (replicate run, ideally also a second model family).
   - Keep the prompts file and hashes, and freeze the sample before labelling.
   - Hold out category until analysis.
5. **Scope note.** Subfield tasks contain almost no teaching, review, administration or writing work (SciNet files those at universal and domain level). Any conclusion about "hidden work" from this design is limited to subfield research tasks.

## Files (Pilot v2)

| File | Status |
|---|---|
| `scripts/run_pipeline_v2.py` | new; stages `prepare`, `label`, `analyze` |
| `data/processed/pilot_v2_sample.csv` | written: the same 50 tasks in v1 order, with substeps. No gap column |
| `data/processed/pilot_v2_prompts.jsonl` | written: 100 checked prompts (main + no_substeps) with SHA-256 |
| `outputs/pilot_v2_validation.json` | written: join and leakage validation |
| `outputs/pilot_v2_summary.txt` | written: prepare-stage summary. `analyze` overwrites it with results |
| `data/processed/openai_observability_v2.csv` | **not created**: needs the API |
| `data/processed/analysis_ready_v2.csv` | **not created**: needs labels |
| `outputs/pilot_v2_v1_comparison.csv` | **not created**: needs labels |

The Pilot v1 files are unchanged; their SHA-256 hashes were re-verified after all v2 runs.

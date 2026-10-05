# Pilot audit: observability and the researcher–paper gap

*Independent methods audit of the 50-task pilot (`observability_v1`, model `gpt-5.6-sol`, seed 42).*
*Everything below was re-computed from `data/raw/` and `data/processed/`. The raw files are byte-identical to the official SciNet release (v1.9.0, github.com/lukasalthoff/scinet).*

**Bottom line.** The labelling stage did not leak the outcome variables. The pilot finds **no association** between LLM-rated paper observability and paper prevalence (r = 0.00, 95% CI −0.28 to 0.28), researcher share, or the gap. The "researcher–paper gap" is not a measure of hidden work. About 70% of its variance comes from paper prevalence alone. The rest reflects two LLM measures with different units of analysis. The original hypothesis is unsupported by this pilot, but the pilot was not designed to test it fairly either.

---

## 1. Data-generating process

Every variable in `analysis_ready.csv` is a **model output**. None is an observation of researcher behaviour.

| Variable | What it is (per SciNet docs) | Unit / denominator | Produced by |
|---|---|---|---|
| `pct_researchers` | One LLM answer per task × subfield: "out of 100 researchers in this subfield, how many perform this task at least occasionally" | researchers, open-ended time window | Claude Opus 5, prompted as "researcher in <subfield> with 10+ years' experience". Tasks were rated in batches per subfield, with prompt anchors pushing ~85% of answers to 90–100 |
| `importance`, `frequency`, `classification` | Same rating call. `classification` = Core if importance ≥ 3 and pct ≥ 67 | task | Claude Opus 5 |
| `prevalence` | `n_involved / n_papers`: share of ~100 sampled papers in the subfield where the task is *stated explicitly or clearly implied* | papers (2000–2020, open access, cited, citation-weighted draw) | Claude Sonnet 5, reading the **first 20,000 characters** of each paper |
| `researcher_paper_gap` | `pct_researchers/100 − prevalence` | mixed units | pipeline |
| `paper_observability` | 0–100: how likely a reader of the final paper (including appendices and supplements) is to detect the task, *given that it was performed* | task, conditional on occurrence | OpenAI `gpt-5.6-sol`, one call per task, no repeats |

Two facts about construction matter for interpretation:

- **Subfield tasks were selected on high participation.** SciNet generated them under an instruction to include only tasks "70%+ of researchers do regularly". Generic teaching, mentoring and admin work was explicitly left to the universal and domain levels.
- **`prevalence` already contains inference.** It counts papers where the task is "clearly implied", not only stated. It is therefore partly an observability-filtered quantity already.

## 2. Pilot design

**Universe.** Subfield-level tasks with a subfield-scope `judged` prevalence row: **4,938 of 6,777** subfield tasks across all 318 subfields.

- 1,188 `expansion` tasks are excluded. They were found in papers and kept only if prevalence ≥ 5%, so excluding them avoids selecting on the outcome.
- 651 tasks have no prevalence row and are also excluded.
- **Verdict on the restriction:** a sound choice for measurement comparability (same judge, same ~100-paper sample, same rating call).

**Sample.** Equal allocation across the 10 categories (target 5 each). Unfilled slots were topped up at random. I reproduced the sample exactly: same 50 `record_id`s in the same order, and every outcome value matches the universe.

**Problems with the design:**

1. **The universe excludes most plausible "hidden work".** Data Analysis and Data Gathering make up 84% of the universe. Mentorship & Teaching and Peer Review & Service have **one task each**, and Writing & Communication has six. Administration, teaching, reviewing, grant writing and similar work lives at the universal and domain levels, which the pilot excludes by design. The pilot therefore tests observability mostly on methods-type tasks that papers are built to report.
2. **The balanced sample is not representative of the universe without weights.** The two n = 1 categories are sampled with certainty, Writing at 5/6 and Design at 5/35, while Data Analysis and Data Gathering are sampled at about 0.4%. Correlations on the sample estimate a category-balanced quantity, not a population one. Unweighted and weighted results are both reported below.
3. **n = 50 limits power.** The minimum detectable |r| at 80% power and α = 0.05 is **≈ 0.39**. The tasks span 45 different subfields, so no within-subfield comparison is possible.
4. **The sampling idea is otherwise fine.** Seeded, frozen, reproducible, and a reasonable way to get breadth for a first look.

## 3. Independence of the OpenAI labelling

**Confirmed, procedurally.** I checked this in four ways:

- I read `build_task_prompt` and `SYSTEM_INSTRUCTION`.
- I rebuilt the exact prompt for sampled rows and string-searched it. None of `pct_researchers`, `prevalence`, `importance`, `frequency`, `classification`, `gap`, "Core", or the rows' numeric values appear.
- `openai_observability.csv` contains no outcome columns.
- Outcomes are joined only afterwards, on `record_id`. The labels in `analysis_ready.csv` match the label file exactly.

The model saw only: domain, field, subfield, **category**, task text, and a substeps block.

**Caveats:**

| Issue | Consequence |
|---|---|
| **Substeps never reached the model.** `substeps.csv.gz` leaves `domain` empty for subfield tasks (documented in SciNet's data README). The pipeline joins on `domain` too, so `substeps_text` is empty for 100% of the universe. Joining on `task, level, field, subfield` matches 100% of tasks. | Every prompt said "(No substeps available.)". This is not leakage, but the model rated a one-sentence task statement, not the intended richer description. |
| **`category` is shown to the model** and is shared with the outcome data. | Category explains **67%** of the variance in observability (R², sample). Observability is largely a category proxy: Administration has mean 37, most others are 85–94. |
| No prompt or response log was kept (`store=False`). Only the code, and response IDs that cannot be retrieved, document what was sent. | Independence is verified from the code and the reproduced prompt, not from a run log. |
| Single pass, default sampling, one model, no repeat or alternative rater. | Reliability of `paper_observability` is unknown. Self-reported confidence tracks the score (ρ = 0.91), so it carries no independent information. |
| Independence is procedural, not epistemic. All four measures come from LLMs trained on overlapping literature. | Shared priors could create or mask associations. |

## 4. Descriptive results (pilot, n = 50, unweighted)

| | mean | sd | median | IQR | min–max |
|---|---|---|---|---|---|
| `paper_observability` | 79.4 | 23.3 | 88 | 72–95 | 15–99 |
| `prevalence` | 0.20 | 0.20 | 0.135 | 0.06–0.29 | 0–0.79 |
| `pct_researchers` | 89.9 | 11.0 | 92 | 86–97 | 40–100 |
| `researcher_paper_gap` | 0.70 | 0.18 | 0.755 | 0.63–0.82 | 0.21–0.93 |
| `frequency` (1–7) | 3.2 | 0.75 | 3 | 3–3.75 | 2–5 |

- Observability is **ceiling-heavy**. 48% of tasks score ≥ 90 and 31 of 50 are classed "explicit". Only six are "mostly_hidden": four Administration maintenance or calibration tasks, one Peer Review task and one Writing task. The weighted mean is 87.
- The gap is **positive in every pilot task** and in 99.9% of the universe.
- The sample's marginal distributions of prevalence, pct_researchers, gap and frequency do not differ from the universe (KS p ≥ 0.38).
- **Category means.** Category explains 67% of observability variance, 28% of prevalence variance, 32% of pct_researchers variance, and 15% of gap variance (adjusted R² −0.04). Two categories have n = 1, so no category-level claim is supported.

## 5. Main correlations

**Pilot sample, n = 50:**

| Pair | Pearson r [95% CI] | Spearman ρ [bootstrap 95% CI] | Weighted r* |
|---|---|---|---|
| observability – prevalence | **0.00** [−0.28, 0.28] | −0.03 [−0.30, 0.25] | −0.25 |
| observability – pct_researchers | −0.12 [−0.39, 0.16] | −0.11 [−0.39, 0.19] | −0.22 |
| observability – gap | −0.07 [−0.35, 0.21] | −0.18 [−0.43, 0.10] | +0.11 |
| observability – frequency | −0.06 [−0.33, 0.22] | −0.04 [−0.31, 0.24] | |
| prevalence – pct_researchers | 0.45 [0.19, 0.64] | 0.64 [0.47, 0.75] | |
| prevalence – frequency | 0.51 [0.27, 0.69] | 0.59 [0.42, 0.73] | |
| **prevalence – gap** | **−0.83** [−0.90, −0.72] | −0.73 | |
| pct_researchers – gap | 0.13 [−0.16, 0.39] | −0.05 | |

\*Inverse-inclusion weights, reweighting the balanced sample back to universe category shares. Driven by a few heavily weighted tasks, so unstable.

Robustness checks for observability:

- Within-category rank correlation with prevalence (categories with n ≥ 5): ρ = 0.00.
- Dropping the two singleton categories: ρ = −0.02.
- OLS of prevalence on observability with category fixed effects: b = 0.001 (HC3 p = 0.76).
- OLS of gap on observability with category fixed effects: b = −0.002 (p = 0.49).

**The same patterns hold in the full universe (n = 4,938, no OpenAI data needed):**

| Pair | Correlation |
|---|---|
| gap – prevalence | −0.81 |
| gap – pct_researchers | 0.13 |
| pct_researchers – prevalence (Spearman) | 0.61 |
| prevalence – frequency (Spearman) | 0.52 |
| prevalence – importance (Spearman) | 0.56 |

Subfield explains about 23% of prevalence variance, while category explains 3%.

## 6. The denominator problem

`pct_researchers` and `prevalence` answer different questions:

- **`pct_researchers`**: of the *researchers* in a subfield, how many ever do the task, "at least occasionally".
- **`prevalence`**: of the *papers* in a subfield, how many show the task.

Even with perfectly observable papers, these diverge for three reasons:

1. **Per-project rarity.** A task every researcher does on one project in ten gives share ≈ 1.0 but prevalence ≈ 0.1. The pilot shows this directly. "Determine enantiomeric excess using chiral HPLC or NMR shift reagents" has observability 96, pct 75 and prevalence 0.10. "Design randomized controlled trials…" has observability 99, pct 80 and prevalence 0.07. Both are tasks a paper would clearly show *if performed*, but most papers in the subfield do not perform them.
2. **Division of labour.** Team papers can contain a task that only one co-author does, which pushes in the opposite direction.
3. **Measurement asymmetries layered on top:**
   - pct_researchers is prompt-anchored near 100 and tasks were pre-selected at a 70% threshold. Its sd is 12 points, against prevalence's 20.
   - prevalence is judged on the first 20k characters, whereas the observability prompt assumes the full paper plus supplements.
   - prevalence already credits "clearly implied" tasks.

Consequences:

- The raw gap is mostly 1 − prevalence. Var(share) = 0.012 vs var(prevalence) = 0.039. It is ~0.7 for almost every task because the two scales sit at opposite ends, not because ~70% of work is hidden.
- **The raw difference must not be interpreted as hidden work.** A low prevalence is better explained by *how often the task occurs per paper* than by *whether a paper would show it*. That is consistent with the null observability result and with prevalence tracking rated frequency (ρ ≈ 0.5).
- Converting between the two units would need papers-per-researcher and per-project occurrence rates. **No researcher-level or paper-level microdata exist in this repository**, so that conversion cannot be done here, and this audit does not propose it.

## 7. Interpretation

**Supported by this pilot:**

- The pipeline is reproducible. The raw data match the SciNet release, and the sample and gap reproduce exactly.
- The OpenAI model was not shown pct_researchers, prevalence, importance, frequency, classification or the gap.
- In this sample, LLM-rated observability shows **no detectable association** with paper prevalence, researcher share or the gap. Moderate-to-large positive associations (r ≳ 0.3–0.4) are unlikely. Small ones cannot be ruled out.
- Observability ratings cluster near the ceiling and mostly differentiate maintenance/administration-type tasks from everything else.
- Across all 4,938 tasks, the gap is driven by prevalence. The two SciNet measures agree moderately on task *ordering* (ρ = 0.61). Prevalence co-varies with rated frequency and importance.

**Not supported:**

- That researchers perform work that papers do not show, or any estimate of how much (for example "≈ 70% hidden").
- That the gap measures invisibility, or that observability "explains" or "fails to explain" hidden work in general. The universe omits generic work, n is small, substeps were missing, and the rater is single and unvalidated.
- Any statement about real researchers. pct_researchers is an LLM estimate, not a survey prevalence.
- Any category-level conclusion. Two categories have n = 1, and Writing has 6 tasks in the whole universe.

## 8. Limitations

- All inputs and the new label are LLM judgments with no human ground truth in the repository.
- `paper_observability`: single run, no test–retest, no second model. Substeps were absent by mistake, and category was given to the model.
- The universe is restricted to subfield tasks and excludes universal and domain tasks (teaching, review, admin, grant writing).
- n = 50 across 45 subfields, category-balanced (needs weights), so power is low.
- Reading frames differ: 20k characters for prevalence vs full paper and supplements for observability.
- The gap is a difference of bounded variables on different units, with a ceiling-compressed minuend.
- No run log or prompt archive. `pipeline_summary.txt` records local Windows paths only.

## 9. Recommended next step

Do not scale the OpenAI labelling yet. First, using only files already in `data/raw/`:

1. **Fix the substeps join.** Key on `task, level, field, subfield`, because `domain` is empty in substeps for subfield tasks.
2. **Drop the raw gap as an outcome.** Instead, analyse **within-subfield concordance** between `pct_researchers` and `prevalence` across all 4,938 tasks:
   - the per-subfield Spearman ρ, giving 318 estimates and their distribution;
   - rank-residuals from a model of prevalence rank on researcher-share rank, frequency and category with subfield fixed effects.
   This costs nothing, uses the full population, and respects the unit difference by comparing orderings rather than levels.
3. **Only then label observability again,** treating it as one candidate predictor of the residuals. Use:
   - substeps in the prompt and category withheld;
   - a proportional or stratified sample with weights, of n ≈ 200 (80% power for r ≈ 0.2);
   - two independent runs, or a second model, to estimate reliability.

## 10. Revised research question

> **Within SciNet subfields, how closely do LLM-rated researcher participation (`pct_researchers`) and LLM-judged paper prevalence agree on the *relative ordering* of tasks? Are the tasks that papers show less often than their participation rank implies distinguished by rated frequency, importance, or category, and, secondarily, by independently rated paper observability?**

This question is answerable with the data in the repository (`tasks.csv`, `task_ratings.csv`, `task_prevalence.csv`, `substeps.csv.gz`):

- It compares rankings within a subfield, so it does not assume the two denominators are on a common scale.
- It treats both measures as model-generated constructs rather than ground truth.
- It treats observability as a hypothesis to test, not a premise.

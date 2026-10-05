"""
What Papers Don't Show — Pilot v2
---------------------------------
Corrected re-run of the 50-task pilot. Pilot v1 (scripts/run_pipeline.py and
its outputs) is kept unchanged as an audited artifact; see
outputs/pilot_v1_MANIFEST.md and outputs/pilot_v1_audit.md.

What changed relative to v1:
1. Substeps join fixed. In substeps.csv.gz, subfield-level rows leave `domain`
   empty, so v1's key (task, level, domain, field, subfield) matched nothing
   and every prompt said "(No substeps available.)". v2 joins subfield tasks
   on (task, level, field, subfield) and fails loudly if the join is incomplete.
2. Category removed from the prompt. The model sees only domain, field,
   subfield, task and substeps. Category stays in the dataset for analysis.
3. Prompt rewritten (observability_v2): explicit / inferable / mostly_hidden
   definitions, substeps used as evidence, no judgment of how common a task is.
4. Every prompt is built from an allowlist of fields, checked for leakage
   before any API call, hashed, and saved.
5. The frozen v1 sample (same 50 tasks, same order) is reused.

Stages:
    python scripts/run_pipeline_v2.py prepare      # no API calls
    python scripts/run_pipeline_v2.py label        # OpenAI labelling (main)
    python scripts/run_pipeline_v2.py label --variant no_substeps   # optional ablation
    python scripts/run_pipeline_v2.py label --variant replicate     # optional repeat run
    python scripts/run_pipeline_v2.py analyze      # v1 vs v2 + exploratory stats

The researcher_paper_gap (pct_researchers/100 - prevalence) is deliberately
not carried into v2: it mixes two denominators and is not an outcome here.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Same model as v1, so that v1 vs v2 differences come from the prompt/inputs.
MODEL = "gpt-5.6-sol"

PROMPT_VERSION = "observability_v2"

RANDOM_SEED = 42

MAX_RETRIES = 4

API_DELAY_SECONDS = 0.25

# The join must match at least this share of comparable tasks.
# SciNet documents only 2 subfield tasks without a breakdown.
MIN_SUBSTEP_MATCH_RATE = 0.995

# Variants: main = full v2 prompt; no_substeps = same prompt with substeps
# withheld (isolates the substeps effect from the category removal);
# replicate = second independent run of the main prompt (run-to-run noise).
VARIANTS = ("main", "no_substeps", "replicate")

# Within-category correlations are only reported for categories this large.
MIN_N_WITHIN_CATEGORY = 8

# A score change of at least this many points counts as "material".
MATERIAL_CHANGE = 15

BOOTSTRAP_DRAWS = 5000


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

TASKS_PATH = RAW_DIR / "tasks.csv"
RATINGS_PATH = RAW_DIR / "task_ratings.csv"
PREVALENCE_PATH = RAW_DIR / "task_prevalence.csv"
SUBSTEPS_PATH = RAW_DIR / "substeps.csv.gz"

# Pilot v1 (read-only).
V1_SAMPLE_PATH = PROCESSED_DIR / "pilot_sample.csv"
V1_ANALYSIS_PATH = PROCESSED_DIR / "analysis_ready.csv"

# Pilot v2.
SAMPLE_PATH = PROCESSED_DIR / "pilot_v2_sample.csv"
PROMPTS_PATH = PROCESSED_DIR / "pilot_v2_prompts.jsonl"
VALIDATION_PATH = OUTPUT_DIR / "pilot_v2_validation.json"
ANALYSIS_READY_PATH = PROCESSED_DIR / "analysis_ready_v2.csv"
COMPARISON_PATH = OUTPUT_DIR / "pilot_v2_v1_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "pilot_v2_summary.txt"


def labels_path(variant: str) -> Path:
    suffix = "" if variant == "main" else f"_{variant}"
    return PROCESSED_DIR / f"openai_observability_v2{suffix}.csv"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# ============================================================
# LOAD SCINET AND BUILD THE COMPARABLE UNIVERSE
# ============================================================

KEY_COLUMNS = ["task", "level", "domain", "field", "subfield"]

# substeps.csv.gz fills `domain` only for domain-level tasks, so subfield tasks
# must be matched without it. (task, field, subfield) is unique among subfield
# tasks; two subfield names exist in two fields, so `field` is required.
SUBSTEP_KEY = ["task", "level", "field", "subfield"]


def read_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False keeps empty hierarchy cells as "" rather than NaN.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for col in KEY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def build_universe() -> tuple[pd.DataFrame, dict]:
    print("\nLoading SciNet...")
    tasks = read_csv(TASKS_PATH)
    ratings = read_csv(RATINGS_PATH)
    prevalence = read_csv(PREVALENCE_PATH)
    substeps = read_csv(SUBSTEPS_PATH)

    tasks_sf = tasks[tasks["level"] == "subfield"]
    ratings_sf = ratings[ratings["level"] == "subfield"]
    prevalence_sf = prevalence[
        (prevalence["level"] == "subfield")
        & (prevalence["scope"] == "subfield")
        & (prevalence["source"] == "judged")
    ]
    substeps_sf = substeps[substeps["level"] == "subfield"]

    for name, df in [("tasks", tasks_sf), ("ratings", ratings_sf), ("prevalence", prevalence_sf)]:
        if df.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"Duplicate composite keys in {name}.")

    base = tasks_sf[KEY_COLUMNS + ["category"]].merge(
        ratings_sf[KEY_COLUMNS + ["importance", "pct_researchers", "frequency", "classification"]],
        on=KEY_COLUMNS, how="inner", validate="one_to_one",
    ).merge(
        prevalence_sf[KEY_COLUMNS + ["n_papers", "n_involved", "prevalence", "se", "source", "judge"]],
        on=KEY_COLUMNS, how="inner", validate="one_to_one",
    )
    base = base.dropna(subset=["pct_researchers", "prevalence", "category", "task"])
    base["record_id"] = base.apply(make_record_id, axis=1)
    if base["record_id"].duplicated().any():
        raise ValueError("record_id is not unique.")

    base, validation = attach_substeps(base, substeps_sf)
    return base, validation


def make_record_id(row: pd.Series) -> str:
    """Identical to v1, so v1 and v2 records can be matched."""
    key = " || ".join(str(row.get(c, "")) for c in KEY_COLUMNS)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ============================================================
# SUBSTEPS JOIN (FIXED) + VALIDATION
# ============================================================

def attach_substeps(base: pd.DataFrame, substeps_sf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if base.duplicated(SUBSTEP_KEY).any():
        raise ValueError(f"{SUBSTEP_KEY} does not uniquely identify comparable tasks.")
    if substeps_sf.duplicated(SUBSTEP_KEY + ["substep_id"]).any():
        raise ValueError("Duplicate substep_id within a task in substeps.csv.gz.")

    steps = substeps_sf.copy()
    steps["_order"] = steps["substep_id"].str.extract(r"(\d+)", expand=False).astype(int)
    steps = steps.sort_values(SUBSTEP_KEY + ["_order"])
    steps["_line"] = steps["substep_id"] + ": " + steps["substep"].astype(str).str.strip()

    grouped = (
        steps.groupby(SUBSTEP_KEY, sort=False)
        .agg(n_substeps=("_line", "size"), substeps_text=("_line", "\n".join))
        .reset_index()
    )

    out = base.merge(grouped, on=SUBSTEP_KEY, how="left", validate="one_to_one")
    out["n_substeps"] = out["n_substeps"].fillna(0).astype(int)
    out["substeps_text"] = out["substeps_text"].fillna("")

    # What the v1 key would have matched, to document the bug.
    v1_keys = steps[KEY_COLUMNS].drop_duplicates()
    v1_matched = int(base[KEY_COLUMNS].merge(v1_keys, on=KEY_COLUMNS).shape[0])

    n = len(out)
    matched = int((out["n_substeps"] > 0).sum())
    validation = {
        "join_key": SUBSTEP_KEY,
        "substeps_domain_filled_for_subfield_rows": int((substeps_sf["domain"] != "").sum()),
        "comparable_tasks": n,
        "tasks_with_matched_substeps": matched,
        "pct_tasks_with_matched_substeps": round(100 * matched / n, 3),
        "tasks_with_zero_substeps": n - matched,
        "zero_substep_tasks": out.loc[out["n_substeps"] == 0, ["field", "subfield", "task"]].to_dict("records"),
        "substeps_per_task": out.loc[out["n_substeps"] > 0, "n_substeps"].describe().round(2).to_dict(),
        "v1_key_matches_for_comparison": v1_matched,
    }

    print("\nSUBSTEPS JOIN VALIDATION")
    for k in ["join_key", "comparable_tasks", "tasks_with_matched_substeps",
              "pct_tasks_with_matched_substeps", "tasks_with_zero_substeps",
              "v1_key_matches_for_comparison"]:
        print(f"  {k}: {validation[k]}")

    if matched / n < MIN_SUBSTEP_MATCH_RATE:
        raise RuntimeError(
            f"Substeps join failed: only {matched}/{n} tasks matched "
            f"(< {MIN_SUBSTEP_MATCH_RATE:.1%}). Inspect the join key."
        )
    return out, validation


# ============================================================
# FROZEN SAMPLE (REUSED FROM V1)
# ============================================================

SAMPLE_COLUMNS = [
    "record_id", "task", "level", "domain", "field", "subfield", "category",
    "importance", "pct_researchers", "frequency", "classification",
    "n_papers", "n_involved", "prevalence", "se", "source", "judge",
    "n_substeps", "substeps_text",
]


def build_sample(universe: pd.DataFrame) -> pd.DataFrame:
    v1 = pd.read_csv(V1_SAMPLE_PATH, dtype={"record_id": str})
    ids = list(v1["record_id"])

    missing = set(ids) - set(universe["record_id"])
    if missing:
        raise RuntimeError(f"{len(missing)} v1 sample tasks are not in the v2 universe: {missing}")

    sample = universe.set_index("record_id").loc[ids].reset_index()[SAMPLE_COLUMNS]

    # Same tasks, same SciNet values as v1.
    check = sample.merge(v1, on="record_id", suffixes=("", "_v1"))
    for col in ["pct_researchers", "prevalence", "importance", "frequency"]:
        if not np.allclose(check[col].astype(float), check[f"{col}_v1"].astype(float)):
            raise RuntimeError(f"{col} differs between v1 sample and v2 universe.")
    for col in ["task", "category", "classification"]:
        if not (check[col] == check[f"{col}_v1"]).all():
            raise RuntimeError(f"{col} differs between v1 sample and v2 universe.")

    zero = sample[sample["n_substeps"] == 0]
    if len(zero):
        raise RuntimeError(f"{len(zero)} sampled tasks have no substeps: {list(zero['record_id'])}")

    if SAMPLE_PATH.exists():
        frozen = pd.read_csv(SAMPLE_PATH, dtype={"record_id": str})
        if list(frozen["record_id"]) != ids:
            raise RuntimeError("Existing pilot_v2_sample.csv does not match the v1 sample order.")
        print(f"\nReusing frozen v2 sample ({len(frozen)} tasks).")
    else:
        sample.to_csv(SAMPLE_PATH, index=False, encoding="utf-8")
        print(f"\nWrote {rel(SAMPLE_PATH)} ({len(sample)} tasks, same order as v1).")

    return sample


# ============================================================
# OPENAI PROMPT (observability_v2)
# ============================================================

# The only record fields that may enter a prompt.
PROMPT_FIELDS = ("domain", "field", "subfield", "task", "substeps_text")

SYSTEM_INSTRUCTION = """
You are assisting a research-methods study of scientific work.

Your job is to judge PAPER OBSERVABILITY for one research task.

Assume the task below WAS performed during a research project, and that the
project resulted in a conventional peer-reviewed journal article. A reader has
access only to that final article and its normal supplementary material
(appendices, supplementary methods, data or code availability statements,
acknowledgements, references).

Question: how likely is a careful reader of that article to detect, or
reasonably infer, that this specific task occurred?

Do NOT judge how common the task is among researchers or papers, how often it
is done, or how valuable, difficult or central it is. A rare task can be highly
observable and a universal task can be invisible. Judge only detectability,
given that the task was performed.

Use the substeps as evidence about what performing the task entails. Consider
which substeps leave traces that articles normally report (methods text,
results, figures, tables, supplementary files, formal statements), which can be
inferred from reported outputs, and which normally leave no trace. Score the
task as a whole, giving most weight to the substeps that make up most of the
work.

Score anchors (0-100):

0:
Essentially invisible from the final paper. A reader would normally have no
way to know it happened.

25:
Mostly hidden. It might occasionally be inferred indirectly.

50:
Sometimes visible or reasonably inferable, but often not.

75:
Usually visible or strongly inferable from the paper.

100:
Explicitly stated, directly documented, or virtually unavoidable to infer
from the published paper.

Visibility class:
- "explicit": normally directly reported in the article or its supplement.
- "inferable": not necessarily stated, but a careful reader can reasonably
  infer that it occurred from what is reported.
- "mostly_hidden": normally not detectable from the published article.

Return ONLY valid JSON with exactly these fields:

{
  "paper_observability": integer from 0 to 100,
  "visibility_class": one of ["explicit", "inferable", "mostly_hidden"],
  "likely_location": one of
      [
        "methods",
        "results",
        "appendix_or_supplement",
        "acknowledgements",
        "references_or_citations",
        "multiple",
        "nowhere"
      ],
  "evidence_substeps": list of up to 5 substep IDs (for example ["S2", "S5"])
      that most determined the score, or [] if no substeps were provided,
  "confidence": integer from 0 to 100,
  "rationale": one concise sentence, maximum 40 words
}
""".strip()

USER_TEMPLATE = """
SCIENTIFIC CONTEXT

Domain:
{domain}

Field:
{field}

Subfield:
{subfield}

Research task:
{task}

What performing this task involves (substeps, in order):
{substeps}

Rate only how observable this task would be from the final published article.
""".strip()

NO_SUBSTEPS_TEXT = "(No substeps provided.)"


def build_prompt(fields: dict, variant: str) -> str:
    """
    Build the exact text sent to OpenAI. Accepts only PROMPT_FIELDS, so
    category and SciNet outcome columns cannot be referenced here.
    """
    if set(fields) != set(PROMPT_FIELDS):
        raise ValueError(f"Prompt fields must be exactly {PROMPT_FIELDS}, got {sorted(fields)}")

    steps = fields["substeps_text"].strip()
    if variant == "no_substeps":
        steps = NO_SUBSTEPS_TEXT
    elif not steps:
        raise ValueError("Empty substeps for a main/replicate prompt.")

    user = USER_TEMPLATE.format(
        domain=fields["domain"], field=fields["field"], subfield=fields["subfield"],
        task=fields["task"], substeps=steps,
    )
    return SYSTEM_INSTRUCTION + "\n\n" + user


def prompt_fields(row: pd.Series) -> dict:
    return {k: str(row[k]) for k in PROMPT_FIELDS}


def inserted_content(fields: dict, variant: str) -> str:
    """The SciNet text that build_prompt actually inserts for this variant."""
    values = [fields["domain"], fields["field"], fields["subfield"], fields["task"]]
    if variant != "no_substeps":
        values.append(fields["substeps_text"])
    return " \n ".join(values)


# ============================================================
# LEAKAGE CHECKS
# ============================================================

# Column identifiers: must never appear anywhere in a prompt.
BANNED_IDENTIFIERS = ["pct_researchers", "researcher_paper_gap", "researcher_share"]

# Ordinary words that are also column names. They must not appear in our own
# template text. SciNet's task and substep text uses some of them as normal
# vocabulary (e.g. "frequency response", "prevalence estimates", "IRB review
# category"), so in a prompt they may occur only inside that SciNet content,
# never in text we add.
BANNED_WORDS = ["prevalence", "importance", "frequency", "classification", "category"]

# A banned name used as a field label, e.g. "Task category:" or "frequency =".
LABEL_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:task[ \t]+)?(?:category|prevalence|importance|frequency|classification"
    r"|pct_researchers|researcher_paper_gap|researcher_share)[ \t]*[:=]"
)

ALLOWED_TEMPLATE_LABELS = {"domain", "field", "subfield", "research task",
                           "what performing this task involves (substeps, in order)"}


def check_template() -> None:
    """Our own text (instruction + user template) contains no banned term."""
    template = (SYSTEM_INSTRUCTION + "\n" + USER_TEMPLATE + "\n" + NO_SUBSTEPS_TEXT).lower()
    for term in BANNED_IDENTIFIERS + BANNED_WORDS:
        assert term not in template, f"Banned term {term!r} appears in the prompt template."

    labels = {m.group(1).strip().lower() for m in re.finditer(r"(?m)^([A-Za-z][^:\n{]*):\s*$", USER_TEMPLATE)}
    assert labels == ALLOWED_TEMPLATE_LABELS, f"Unexpected template labels: {labels}"


def check_prompt(prompt: str, content: str, record: pd.Series) -> dict:
    """
    Assert that one prompt leaks nothing. Returns the occurrences of banned
    words that come from SciNet content, for the audit trail.
    """
    low = prompt.lower()
    content = content.lower()

    for term in BANNED_IDENTIFIERS:
        assert term not in low, f"{record['record_id']}: identifier {term!r} in prompt."

    assert not LABEL_PATTERN.search(prompt), (
        f"{record['record_id']}: banned name used as a labeled field: "
        f"{LABEL_PATTERN.search(prompt).group(0)!r}"
    )

    content_hits = {}
    for term in BANNED_WORDS:
        n_prompt, n_content = low.count(term), content.count(term)
        assert n_prompt == n_content, (
            f"{record['record_id']}: {term!r} occurs {n_prompt}x in prompt but "
            f"{n_content}x in SciNet content; the extra occurrences come from our own text."
        )
        if n_content:
            content_hits[term] = n_content

    # The record's category label may not be added by us either.
    cat = str(record["category"]).lower()
    assert low.count(cat) == content.count(cat), (
        f"{record['record_id']}: category value {record['category']!r} added to prompt."
    )

    return content_hits


def self_test_checks(row: pd.Series) -> None:
    """Prove the checks are not vacuous: injected leaks must be caught."""
    fields = prompt_fields(row)
    content = inserted_content(fields, "main")
    clean = build_prompt(fields, "main")
    injections = [
        clean + f"\n\nTask category:\n{row['category']}",
        clean + f"\nprevalence: {row['prevalence']}",
        clean + f"\npct_researchers = {row['pct_researchers']}",
        clean + "\nNote: researcher_paper_gap is large.",
        clean + "\nThe frequency of this task is high.",
    ]
    for leaked in injections:
        try:
            check_prompt(leaked, content, row)
        except AssertionError:
            continue
        raise RuntimeError(f"Leakage check failed to catch an injected leak:\n{leaked[-120:]}")

    try:
        build_prompt({**fields, "category": row["category"]}, "main")
    except ValueError:
        pass
    else:
        raise RuntimeError("build_prompt accepted a non-allowlisted field.")


def build_and_check_prompts(sample: pd.DataFrame) -> pd.DataFrame:
    check_template()
    self_test_checks(sample.iloc[0])

    records = []
    for variant in ("main", "no_substeps"):
        for _, row in sample.iterrows():
            fields = prompt_fields(row)
            prompt = build_prompt(fields, variant)
            hits = check_prompt(prompt, inserted_content(fields, variant), row)
            records.append({
                "record_id": row["record_id"],
                "variant": variant,
                "prompt_version": PROMPT_VERSION,
                "model": MODEL,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "n_substeps_in_prompt": 0 if variant == "no_substeps" else int(row["n_substeps"]),
                "banned_words_from_scinet_content": hits,
                "prompt": prompt,
            })

    with PROMPTS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    prompts = pd.DataFrame(records)
    n_hit = prompts.loc[prompts["variant"] == "main", "banned_words_from_scinet_content"].map(bool).sum()
    print(f"\nLeakage checks passed for {len(prompts)} prompts "
          f"({len(sample)} tasks x 2 variants). Saved to {rel(PROMPTS_PATH)}.")
    print(f"Tasks whose own SciNet text contains a column-name word: {n_hit}")
    return prompts


# ============================================================
# STAGE: PREPARE
# ============================================================

def prepare() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    universe, validation = build_universe()
    sample = build_sample(universe)
    prompts = build_and_check_prompts(sample)

    main = prompts[prompts["variant"] == "main"]
    validation.update({
        "sample_tasks": len(sample),
        "sample_tasks_with_substeps": int((sample["n_substeps"] > 0).sum()),
        "sample_substeps_per_task": sample["n_substeps"].describe().round(2).to_dict(),
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "prompts_checked": len(prompts),
        "prompt_fields_allowlist": list(PROMPT_FIELDS),
        "leakage_self_test": "passed",
        "tasks_with_column_name_words_in_scinet_text": {
            r["record_id"]: r["banned_words_from_scinet_content"]
            for _, r in main.iterrows() if r["banned_words_from_scinet_content"]
        },
        "main_prompt_chars": main["prompt"].str.len().describe().round(0).to_dict(),
    })
    VALIDATION_PATH.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    return sample, prompts, validation


# ============================================================
# STAGE: LABEL
# ============================================================

REQUIRED_FIELDS = {"paper_observability", "visibility_class", "likely_location",
                   "evidence_substeps", "confidence", "rationale"}
ALLOWED_VISIBILITY = {"explicit", "inferable", "mostly_hidden"}
ALLOWED_LOCATIONS = {"methods", "results", "appendix_or_supplement", "acknowledgements",
                     "references_or_citations", "multiple", "nowhere"}


def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_response(raw_text: str, n_substeps: int) -> dict:
    parsed = json.loads(clean_json_text(raw_text))
    if set(parsed) != REQUIRED_FIELDS:
        raise ValueError(f"JSON fields do not match schema: {sorted(parsed)}")

    obs, conf = int(parsed["paper_observability"]), int(parsed["confidence"])
    if not 0 <= obs <= 100 or not 0 <= conf <= 100:
        raise ValueError("Score outside 0-100.")
    if parsed["visibility_class"] not in ALLOWED_VISIBILITY:
        raise ValueError("Invalid visibility_class.")
    if parsed["likely_location"] not in ALLOWED_LOCATIONS:
        raise ValueError("Invalid likely_location.")

    ev = parsed["evidence_substeps"]
    valid_ids = {f"S{i}" for i in range(1, n_substeps + 1)}
    if not isinstance(ev, list) or len(ev) > 5 or any(e not in valid_ids for e in ev):
        raise ValueError(f"Invalid evidence_substeps: {ev}")

    parsed["paper_observability"], parsed["confidence"] = obs, conf
    return parsed


def call_openai(client, prompt: str, n_substeps: int) -> tuple[dict, str, str]:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Same call shape as v1: one user message, nothing stored server-side.
            response = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
                store=False,
            )
            raw = response.output_text
            return parse_response(raw, n_substeps), raw, getattr(response, "id", "")
        except Exception as exc:
            last_error = exc
            print(f"  API attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} attempts: {last_error}")


def label(variant: str) -> pd.DataFrame:
    sample, prompts, _ = prepare()

    prompt_variant = "main" if variant == "replicate" else variant
    prompts = prompts[prompts["variant"] == prompt_variant].set_index("record_id")

    # Re-verify that the saved, checked prompt is byte-identical to what is sent.
    saved = {}
    with PROMPTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["variant"] == prompt_variant:
                saved[r["record_id"]] = r["prompt_sha256"]

    out_path = labels_path(variant)
    done = pd.read_csv(out_path, dtype={"record_id": str}) if out_path.exists() else pd.DataFrame()
    done_ids = set(done["record_id"]) if not done.empty else set()
    todo = sample[~sample["record_id"].isin(done_ids)]
    print(f"\nVariant {variant!r}: {len(done_ids)} labeled, {len(todo)} to go -> {rel(out_path)}")
    if todo.empty:
        return done

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and sys.stdin.isatty():
        api_key = getpass.getpass("Paste your OpenAI API key (input will be hidden): ")
    if not api_key or not api_key.strip():
        raise RuntimeError("No OpenAI API key. Set OPENAI_API_KEY. Nothing was sent.")

    from openai import OpenAI  # imported here so prepare/analyze run without it
    client = OpenAI(api_key=api_key)

    results = done.to_dict("records") if not done.empty else []
    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        rid = row["record_id"]
        prompt = prompts.loc[rid, "prompt"]
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert sha == saved[rid] == prompts.loc[rid, "prompt_sha256"], f"{rid}: prompt changed after checks."
        check_prompt(prompt, inserted_content(prompt_fields(row), prompt_variant), row)

        n_steps = int(prompts.loc[rid, "n_substeps_in_prompt"])
        print(f"[{i}/{len(todo)}] {row['subfield']} | {row['task'][:70]}")
        parsed, raw, response_id = call_openai(client, prompt, n_steps)
        print(f"  -> {parsed['paper_observability']} ({parsed['visibility_class']})")

        results.append({
            "record_id": rid,
            "variant": variant,
            "task": row["task"],
            "domain": row["domain"],
            "field": row["field"],
            "subfield": row["subfield"],
            "paper_observability": parsed["paper_observability"],
            "visibility_class": parsed["visibility_class"],
            "likely_location": parsed["likely_location"],
            "evidence_substeps": json.dumps(parsed["evidence_substeps"]),
            "confidence": parsed["confidence"],
            "rationale": parsed["rationale"],
            "raw_output": raw,
            "n_substeps_in_prompt": n_steps,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": sha,
            "api_response_id": response_id,
            "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        # Checkpoint after every successful call.
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8")
        time.sleep(API_DELAY_SECONDS)

    return pd.DataFrame(results)


# ============================================================
# STATISTICS HELPERS (descriptive only; no significance tests)
# ============================================================

def average_ranks(a: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(a))
    ranks[a.argsort(kind="mergesort")] = np.arange(1, len(a) + 1)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=ranks) / counts)[inv]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(average_ranks(np.asarray(x, float)), average_ranks(np.asarray(y, float)))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn, seed: int = RANDOM_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            draws.append(fn(x[i], y[i]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return round(float(lo), 2), round(float(hi), 2)


def corr_row(df: pd.DataFrame, a: str, b: str, ci: bool = True) -> dict:
    d = df[[a, b]].dropna().astype(float)
    x, y = d[a].to_numpy(), d[b].to_numpy()
    row = {"x": a, "y": b, "n": len(d), "pearson": round(pearson(x, y), 3), "spearman": round(spearman(x, y), 3)}
    if ci:
        row["pearson_95ci"] = bootstrap_ci(x, y, pearson)
        row["spearman_95ci"] = bootstrap_ci(x, y, spearman)
    return row


def describe_scores(s: pd.Series) -> dict:
    return {
        "n": int(s.count()), "mean": round(s.mean(), 2), "median": float(s.median()),
        "sd": round(s.std(), 2), "q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75)),
        "min": int(s.min()), "max": int(s.max()),
        "share_ge_90": round((s >= 90).mean(), 3), "share_lt_50": round((s < 50).mean(), 3),
    }


SCORE_BINS = [-1, 24, 49, 74, 89, 100]
SCORE_BIN_LABELS = ["0-24", "25-49", "50-74", "75-89", "90-100"]


# ============================================================
# STAGE: ANALYZE
# ============================================================

def load_labels(variant: str) -> pd.DataFrame | None:
    path = labels_path(variant)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"record_id": str})
    return df


def analyze() -> None:
    sample = pd.read_csv(SAMPLE_PATH, dtype={"record_id": str})
    v2 = load_labels("main")
    if v2 is None or v2["paper_observability"].notna().sum() < len(sample):
        raise RuntimeError(f"Main v2 labels incomplete or missing: {rel(labels_path('main'))}")

    label_cols = ["record_id", "paper_observability", "visibility_class", "likely_location",
                  "evidence_substeps", "confidence", "rationale", "n_substeps_in_prompt",
                  "model", "prompt_version", "prompt_sha256", "api_response_id", "labeled_at_utc"]
    analysis = sample.merge(v2[label_cols], on="record_id", how="left", validate="one_to_one")
    analysis.to_csv(ANALYSIS_READY_PATH, index=False, encoding="utf-8")

    v1 = pd.read_csv(V1_ANALYSIS_PATH, dtype={"record_id": str})
    v1 = v1[["record_id", "paper_observability", "visibility_class", "rationale"]].rename(
        columns={"paper_observability": "obs_v1", "visibility_class": "class_v1", "rationale": "rationale_v1"})

    cmp_ = analysis.merge(v1, on="record_id", how="inner", validate="one_to_one").rename(
        columns={"paper_observability": "obs_v2", "visibility_class": "class_v2", "rationale": "rationale_v2"})
    cmp_["delta_v2_minus_v1"] = cmp_["obs_v2"] - cmp_["obs_v1"]

    extras = {}
    for variant in ("no_substeps", "replicate"):
        lab = load_labels(variant)
        if lab is not None and lab["paper_observability"].notna().sum() == len(sample):
            cmp_ = cmp_.merge(lab[["record_id", "paper_observability"]].rename(
                columns={"paper_observability": f"obs_v2_{variant}"}), on="record_id", how="left")
            extras[variant] = True
    if "no_substeps" in extras:
        cmp_["delta_substeps"] = cmp_["obs_v2"] - cmp_["obs_v2_no_substeps"]
        cmp_["delta_prompt_and_category"] = cmp_["obs_v2_no_substeps"] - cmp_["obs_v1"]
    if "replicate" in extras:
        cmp_["delta_replicate"] = cmp_["obs_v2"] - cmp_["obs_v2_replicate"]

    keep = [c for c in ["record_id", "category", "field", "subfield", "task", "n_substeps",
                        "obs_v1", "obs_v2", "obs_v2_no_substeps", "obs_v2_replicate",
                        "delta_v2_minus_v1", "delta_substeps", "delta_prompt_and_category",
                        "delta_replicate", "class_v1", "class_v2", "evidence_substeps",
                        "rationale_v1", "rationale_v2"] if c in cmp_.columns]
    cmp_[keep].to_csv(COMPARISON_PATH, index=False, encoding="utf-8")

    write_summary(analysis, cmp_, extras)
    print(f"\nWrote {rel(ANALYSIS_READY_PATH)}, {rel(COMPARISON_PATH)}, {rel(SUMMARY_PATH)}")


def substep_lines(row: pd.Series) -> list[str]:
    steps = dict(line.split(": ", 1) for line in str(row["substeps_text"]).split("\n") if ": " in line)
    ids = json.loads(row["evidence_substeps"]) if isinstance(row["evidence_substeps"], str) else []
    return [f"      {i}: {steps.get(i, '?')[:160]}" for i in ids[:3]]


def write_summary(analysis: pd.DataFrame, cmp_: pd.DataFrame, extras: dict) -> None:
    L = []
    add = L.append
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))

    add("WHAT PAPERS DON'T SHOW - PILOT v2 SUMMARY")
    add("=" * 64)
    add(f"Model: {MODEL}   Prompt version: {PROMPT_VERSION}   Sample: frozen v1 sample (n={len(analysis)})")
    add("Exploratory pilot. n = 50: no statistical significance is claimed.")
    add("researcher_paper_gap is not used: it mixes per-researcher and per-paper denominators.")

    add("\n1. SUBSTEPS JOIN VALIDATION")
    for k in ["join_key", "comparable_tasks", "tasks_with_matched_substeps",
              "pct_tasks_with_matched_substeps", "tasks_with_zero_substeps",
              "v1_key_matches_for_comparison", "sample_tasks_with_substeps"]:
        add(f"  {k}: {validation[k]}")

    add("\n2. LEAKAGE CHECKS")
    add(f"  Prompt fields allowlist: {validation['prompt_fields_allowlist']}")
    add(f"  Prompts checked: {validation['prompts_checked']}; self-test: {validation['leakage_self_test']}")
    add(f"  Tasks whose SciNet text itself contains a column-name word: "
        f"{len(validation['tasks_with_column_name_words_in_scinet_text'])}")

    add("\n3. V2 OBSERVABILITY DISTRIBUTION")
    obs = analysis["paper_observability"]
    for k, v in describe_scores(obs).items():
        add(f"  {k}: {v}")
    add("  visibility_class: " + str(analysis["visibility_class"].value_counts().to_dict()))
    add("  likely_location: " + str(analysis["likely_location"].value_counts().to_dict()))
    ev = analysis["evidence_substeps"].map(lambda s: len(json.loads(s)))
    add(f"  tasks citing >=1 evidence substep: {(ev > 0).sum()}/{len(ev)}")
    add(f"  spearman(score, confidence): {spearman(obs.values, analysis['confidence'].values):.2f}")

    add("\n4. V1 vs V2")
    d1, d2 = describe_scores(cmp_["obs_v1"]), describe_scores(cmp_["obs_v2"])
    add(f"  {'':14s}{'v1':>9s}{'v2':>9s}")
    for k in d1:
        add(f"  {k:14s}{d1[k]!s:>9s}{d2[k]!s:>9s}")
    r = corr_row(cmp_, "obs_v1", "obs_v2")
    add(f"  v1~v2 pearson {r['pearson']} {r['pearson_95ci']}, spearman {r['spearman']} {r['spearman_95ci']}")
    d = cmp_["delta_v2_minus_v1"]
    add(f"  delta (v2-v1): mean {d.mean():.2f}, median {d.median():.1f}, mean |delta| {d.abs().mean():.2f}, "
        f"|delta|>={MATERIAL_CHANGE}: {(d.abs() >= MATERIAL_CHANGE).sum()}")
    bins = pd.DataFrame({
        "v1": pd.cut(cmp_["obs_v1"], SCORE_BINS, labels=SCORE_BIN_LABELS).value_counts().sort_index(),
        "v2": pd.cut(cmp_["obs_v2"], SCORE_BINS, labels=SCORE_BIN_LABELS).value_counts().sort_index(),
    })
    add("  score bins:\n" + bins.to_string())
    add("  visibility class v1 (rows) x v2 (cols):\n" + pd.crosstab(cmp_["class_v1"], cmp_["class_v2"]).to_string())
    bycat = cmp_.groupby("category").agg(n=("obs_v1", "size"), v1=("obs_v1", "mean"),
                                         v2=("obs_v2", "mean"), delta=("delta_v2_minus_v1", "mean")).round(1)
    add("  mean score by category (category was shown in v1, hidden in v2):\n" + bycat.to_string())
    for var in ("no_substeps", "replicate"):
        col = f"obs_v2_{var}"
        if col in cmp_:
            rr = corr_row(cmp_, "obs_v2", col)
            add(f"  v2 vs v2_{var}: spearman {rr['spearman']} {rr['spearman_95ci']}, "
                f"mean |diff| {(cmp_['obs_v2'] - cmp_[col]).abs().mean():.2f}")
    if "replicate" not in extras:
        add("  NOTE: no replicate run, so run-to-run noise is not separated from real change.")
    if "no_substeps" not in extras:
        add("  NOTE: no no_substeps run, so the substeps effect is confounded with removing category "
            "and the new prompt wording.")

    add(f"\n5. LARGEST SCORE CHANGES (top 10 by |v2 - v1|)")
    top = cmp_.reindex(cmp_["delta_v2_minus_v1"].abs().sort_values(ascending=False).index).head(10)
    for _, row in top.iterrows():
        add(f"  [{row['obs_v1']:>3} -> {row['obs_v2']:>3}] ({row['category']}) {row['task'][:90]}")
        add(f"      v1: {row['rationale_v1']}")
        add(f"      v2: {row['rationale_v2']}")

    add("\n6. EXAMPLES WHERE SUBSTEPS CHANGED THE SCORE")
    if "delta_substeps" in cmp_:
        add(f"  Attributable: |v2 - v2_no_substeps| >= {MATERIAL_CHANGE} (same prompt, substeps withheld).")
        ex = cmp_[cmp_["delta_substeps"].abs() >= MATERIAL_CHANGE]
    else:
        add(f"  Candidates only: |v2 - v1| >= {MATERIAL_CHANGE} and v2 cites evidence substeps. "
            "Not attributable to substeps alone (see notes in section 4).")
        ex = cmp_[(cmp_["delta_v2_minus_v1"].abs() >= MATERIAL_CHANGE)
                  & (cmp_["evidence_substeps"].map(lambda s: len(json.loads(s))) > 0)]
    if ex.empty:
        add("  None.")
    for _, row in ex.iterrows():
        extra = f", no_substeps {row['obs_v2_no_substeps']}" if "obs_v2_no_substeps" in row else ""
        add(f"  [v1 {row['obs_v1']}, v2 {row['obs_v2']}{extra}] {row['task'][:90]}")
        add(f"      v2: {row['rationale_v2']}")
        L.extend(substep_lines(row))

    add("\n7. EXPLORATORY CORRELATIONS (bootstrap 95% CIs; descriptive only)")
    pairs = [("paper_observability", "prevalence"), ("paper_observability", "pct_researchers"),
             ("paper_observability", "frequency"), ("paper_observability", "importance"),
             ("prevalence", "frequency"), ("prevalence", "importance")]
    tbl = pd.DataFrame([corr_row(analysis, a, b) for a, b in pairs])
    add(tbl.to_string(index=False))

    add(f"\n  Within categories with n >= {MIN_N_WITHIN_CATEGORY}:")
    counts = analysis["category"].value_counts()
    for cat, n in counts.items():
        if n < MIN_N_WITHIN_CATEGORY:
            continue
        sub = analysis[analysis["category"] == cat]
        rows = pd.DataFrame([corr_row(sub, a, b, ci=False) for a, b in pairs])
        add(f"  {cat} (n={n}):\n" + rows.to_string(index=False))
    skipped = counts[counts < MIN_N_WITHIN_CATEGORY]
    add(f"  Not computed (n < {MIN_N_WITHIN_CATEGORY}): " + ", ".join(f"{c} ({n})" for c, n in skipped.items()))

    add("\nOutput files:")
    for p in [SAMPLE_PATH, PROMPTS_PATH, labels_path("main"), ANALYSIS_READY_PATH, COMPARISON_PATH, VALIDATION_PATH]:
        add(f"- {rel(p)}")

    SUMMARY_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_prepare_summary(validation: dict) -> None:
    """Summary written after `prepare`, before any labels exist."""
    L = ["WHAT PAPERS DON'T SHOW - PILOT v2 SUMMARY", "=" * 64,
         f"Model: {MODEL}   Prompt version: {PROMPT_VERSION}",
         "STATUS: prepared. OpenAI labelling has NOT been run; no observability results yet.",
         "", "1. SUBSTEPS JOIN VALIDATION"]
    for k in ["join_key", "substeps_domain_filled_for_subfield_rows", "comparable_tasks",
              "tasks_with_matched_substeps", "pct_tasks_with_matched_substeps",
              "tasks_with_zero_substeps", "v1_key_matches_for_comparison",
              "sample_tasks", "sample_tasks_with_substeps"]:
        L.append(f"  {k}: {validation[k]}")
    L.append(f"  zero-substep tasks: {validation['zero_substep_tasks']}")
    L.append(f"  substeps per sampled task: {validation['sample_substeps_per_task']}")
    L += ["", "2. LEAKAGE CHECKS (all passed)",
          f"  Prompt fields allowlist: {validation['prompt_fields_allowlist']}",
          f"  Prompts built and checked: {validation['prompts_checked']} (50 main + 50 no_substeps)",
          f"  Injected-leak self-test: {validation['leakage_self_test']}",
          "  Template contains none of: " + ", ".join(BANNED_IDENTIFIERS + BANNED_WORDS),
          "  Tasks whose own SciNet text contains a column-name word (allowed, logged):"]
    for rid, hits in validation["tasks_with_column_name_words_in_scinet_text"].items():
        L.append(f"    {rid}: {hits}")
    L += ["", f"Prompts: {rel(PROMPTS_PATH)}", f"Validation: {rel(VALIDATION_PATH)}",
          f"Sample: {rel(SAMPLE_PATH)}"]
    SUMMARY_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "label", "analyze", "all"])
    parser.add_argument("--variant", choices=VARIANTS, default="main")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "prepare":
        _, _, validation = prepare()
        write_prepare_summary(validation)
        print(f"\nWrote {rel(SUMMARY_PATH)}. No API calls were made.")
    elif args.stage == "label":
        label(args.variant)
    elif args.stage == "analyze":
        analyze()
    else:
        label("main")
        analyze()


if __name__ == "__main__":
    main()

"""
What Papers Don't Show
----------------------
Mini research pipeline using SciNet + OpenAI.

Pipeline:
1. Read SciNet source files
2. Restrict to comparable subfield-level tasks
3. Merge task metadata, ratings, prevalence, and substeps
4. Draw a reproducible category-balanced pilot sample
5. Ask OpenAI to independently rate paper observability
6. Save labels incrementally
7. Create an analysis-ready dataset

IMPORTANT:
The OpenAI model NEVER sees pct_researchers or prevalence.
Those variables are merged back only after OpenAI labeling.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gpt-5.6-sol"

# Start small. With ~10 SciNet categories, 50 usually gives
# approximately 5 tasks per category.
PILOT_SAMPLE_SIZE = 50

RANDOM_SEED = 42

PROMPT_VERSION = "observability_v1"

MAX_RETRIES = 4

# Prevent very long substep lists from making prompts unnecessarily large.
MAX_SUBSTEPS = 15

# Small delay between API calls.
API_DELAY_SECONDS = 0.25


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS_PATH = RAW_DIR / "tasks.csv"
RATINGS_PATH = RAW_DIR / "task_ratings.csv"
PREVALENCE_PATH = RAW_DIR / "task_prevalence.csv"
SUBSTEPS_PATH = RAW_DIR / "substeps.csv.gz"

SAMPLE_PATH = PROCESSED_DIR / "pilot_sample.csv"
LABELS_PATH = PROCESSED_DIR / "openai_observability.csv"
ANALYSIS_READY_PATH = PROCESSED_DIR / "analysis_ready.csv"
SUMMARY_PATH = OUTPUT_DIR / "pipeline_summary.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

KEY_COLUMNS = [
    "task",
    "level",
    "domain",
    "field",
    "subfield",
]


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make merge keys consistent.

    Pandas normally reads empty CSV cells as NaN. Replacing them with ""
    makes composite-key merges more predictable.
    """
    df = df.copy()

    for col in KEY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def make_record_id(row: pd.Series) -> str:
    """
    Create our own stable task identifier because SciNet does not
    provide a task_id column.
    """
    key = " || ".join(
        [
            str(row.get("task", "")),
            str(row.get("level", "")),
            str(row.get("domain", "")),
            str(row.get("field", "")),
            str(row.get("subfield", "")),
        ]
    )

    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def validate_required_files() -> None:
    required = [
        TASKS_PATH,
        RATINGS_PATH,
        PREVALENCE_PATH,
        SUBSTEPS_PATH,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "\nMissing required SciNet files:\n"
            + "\n".join(f"  - {x}" for x in missing)
            + "\n\nExpected location: data/raw/"
        )


def validate_columns(
    df: pd.DataFrame,
    required: list[str],
    dataset_name: str,
) -> None:
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def clean_json_text(text: str) -> str:
    """
    Remove accidental Markdown code fences before json.loads().
    """
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


# ============================================================
# LOAD SCINET
# ============================================================

def load_scinet():
    print("\nLoading SciNet datasets...")

    validate_required_files()

    tasks = pd.read_csv(TASKS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    prevalence = pd.read_csv(PREVALENCE_PATH)
    substeps = pd.read_csv(SUBSTEPS_PATH)

    tasks = normalize_key_columns(tasks)
    ratings = normalize_key_columns(ratings)
    prevalence = normalize_key_columns(prevalence)
    substeps = normalize_key_columns(substeps)

    validate_columns(
        tasks,
        [
            "task",
            "category",
            "level",
            "domain",
            "field",
            "subfield",
        ],
        "tasks.csv",
    )

    validate_columns(
        ratings,
        [
            "task",
            "level",
            "domain",
            "field",
            "subfield",
            "importance",
            "pct_researchers",
            "frequency",
            "classification",
        ],
        "task_ratings.csv",
    )

    validate_columns(
        prevalence,
        [
            "task",
            "level",
            "scope",
            "domain",
            "field",
            "subfield",
            "n_papers",
            "n_involved",
            "prevalence",
            "source",
        ],
        "task_prevalence.csv",
    )

    validate_columns(
        substeps,
        [
            "task",
            "level",
            "domain",
            "field",
            "subfield",
            "substep_id",
            "substep",
        ],
        "substeps.csv.gz",
    )

    print(f"tasks.csv:           {tasks.shape}")
    print(f"task_ratings.csv:    {ratings.shape}")
    print(f"task_prevalence.csv: {prevalence.shape}")
    print(f"substeps.csv.gz:     {substeps.shape}")

    return tasks, ratings, prevalence, substeps


# ============================================================
# BUILD COMPARABLE DATASET
# ============================================================

def build_comparable_dataset(
    tasks: pd.DataFrame,
    ratings: pd.DataFrame,
    prevalence: pd.DataFrame,
    substeps: pd.DataFrame,
) -> pd.DataFrame:

    print("\nBuilding comparable subfield-level dataset...")

    # --------------------------------------------------------
    # Restrict to subfield tasks.
    #
    # This avoids mixing aggregation levels.
    # --------------------------------------------------------

    tasks_sf = tasks[tasks["level"] == "subfield"].copy()

    ratings_sf = ratings[
        ratings["level"] == "subfield"
    ].copy()

    prevalence_sf = prevalence[
        (prevalence["level"] == "subfield")
        & (prevalence["scope"] == "subfield")
        & (prevalence["source"] == "judged")
    ].copy()

    substeps_sf = substeps[
        substeps["level"] == "subfield"
    ].copy()

    print(f"Subfield tasks:                    {len(tasks_sf):,}")
    print(f"Subfield ratings:                  {len(ratings_sf):,}")
    print(
        "Subfield judged prevalence rows:   "
        f"{len(prevalence_sf):,}"
    )

    # --------------------------------------------------------
    # Check uniqueness before merging.
    # --------------------------------------------------------

    for name, df in [
        ("tasks", tasks_sf),
        ("ratings", ratings_sf),
        ("prevalence", prevalence_sf),
    ]:
        duplicate_count = df.duplicated(KEY_COLUMNS).sum()

        print(
            f"Duplicate composite keys in {name}: "
            f"{duplicate_count:,}"
        )

        if duplicate_count > 0:
            raise ValueError(
                f"Unexpected duplicate task keys in {name}. "
                "Stop and inspect before continuing."
            )

    # --------------------------------------------------------
    # Aggregate substeps into an ordered text block.
    # --------------------------------------------------------

    def aggregate_steps(group):
        group = group.copy()

        # Extract numeric part from S1, S2, ...
        group["_order"] = (
            group["substep_id"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna("999999")
            .astype(int)
        )

        group = group.sort_values("_order")

        steps = []

        for _, row in group.head(MAX_SUBSTEPS).iterrows():
            steps.append(
                f"{row['substep_id']}: {row['substep']}"
            )

        return "\n".join(steps)

    step_text = (
        substeps_sf.groupby(KEY_COLUMNS, dropna=False)
        .apply(aggregate_steps)
        .reset_index(name="substeps_text")
    )

    # --------------------------------------------------------
    # Merge task metadata + ratings.
    # --------------------------------------------------------

    base = tasks_sf[
        KEY_COLUMNS + ["category"]
    ].merge(
        ratings_sf[
            KEY_COLUMNS
            + [
                "importance",
                "pct_researchers",
                "frequency",
                "classification",
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge paper prevalence.
    # --------------------------------------------------------

    base = base.merge(
        prevalence_sf[
            KEY_COLUMNS
            + [
                "n_papers",
                "n_involved",
                "prevalence",
                "se",
                "source",
                "judge",
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge substeps.
    # --------------------------------------------------------

    base = base.merge(
        step_text,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    base["substeps_text"] = (
        base["substeps_text"].fillna("")
    )

    base["record_id"] = base.apply(
        make_record_id,
        axis=1,
    )

    # --------------------------------------------------------
    # Basic validity checks.
    # --------------------------------------------------------

    if base["record_id"].duplicated().any():
        raise ValueError(
            "Generated record_id is unexpectedly non-unique."
        )

    base["pct_researchers"] = pd.to_numeric(
        base["pct_researchers"],
        errors="coerce",
    )

    base["prevalence"] = pd.to_numeric(
        base["prevalence"],
        errors="coerce",
    )

    base = base.dropna(
        subset=[
            "pct_researchers",
            "prevalence",
            "category",
            "task",
        ]
    )

    # Convert pct_researchers from 0-100 to 0-1.
    base["researcher_share"] = (
        base["pct_researchers"] / 100.0
    )

    # Difference we eventually want Claude / statistical analysis
    # to investigate.
    #
    # IMPORTANT: this variable is NOT sent to OpenAI.
    base["researcher_paper_gap"] = (
        base["researcher_share"]
        - base["prevalence"]
    )

    print(
        "\nComparable tasks after merge: "
        f"{len(base):,}"
    )

    print("\nCategories:")
    print(
        base["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return base


# ============================================================
# BALANCED PILOT SAMPLE
# ============================================================

def balanced_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:

    if n > len(df):
        n = len(df)

    categories = sorted(
        df["category"].dropna().unique()
    )

    if not categories:
        raise ValueError("No task categories found.")

    base_per_category = n // len(categories)
    remainder = n % len(categories)

    sampled_parts = []

    for i, category in enumerate(categories):

        category_df = df[
            df["category"] == category
        ]

        target = (
            base_per_category
            + (1 if i < remainder else 0)
        )

        target = min(
            target,
            len(category_df),
        )

        if target > 0:
            sampled_parts.append(
                category_df.sample(
                    n=target,
                    random_state=seed + i,
                )
            )

    sample = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    # If a small category caused under-filling,
    # fill remaining slots from unsampled tasks.
    if len(sample) < n:

        remaining_ids = set(
            df["record_id"]
        ) - set(
            sample["record_id"]
        )

        remaining = df[
            df["record_id"].isin(remaining_ids)
        ]

        needed = min(
            n - len(sample),
            len(remaining),
        )

        if needed > 0:
            extra = remaining.sample(
                n=needed,
                random_state=seed + 999,
            )

            sample = pd.concat(
                [sample, extra],
                ignore_index=True,
            )

    sample = sample.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    print(
        f"\nPilot sample created: {len(sample)} tasks"
    )

    print("\nPilot category distribution:")
    print(
        sample["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return sample


# ============================================================
# OPENAI PROMPT
# ============================================================

SYSTEM_INSTRUCTION = """
You are helping with a research audit about scientific work.

Your task is NOT to estimate how common, important, difficult, or valuable
a research task is.

Your task is to estimate PAPER OBSERVABILITY:

Assume that the research task really was performed during a research project
that resulted in a conventional peer-reviewed academic article.

Imagine that you only have access to the final published paper and its normal
appendices or supplementary materials.

How likely is a careful reader to detect or reasonably infer that this
specific task occurred?

Do not use outside knowledge about how frequently researchers perform the task.
Do not estimate task prevalence.

Use these anchors:

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

Return ONLY valid JSON with exactly these fields:

{
  "paper_observability": integer from 0 to 100,
  "visibility_class": one of
      ["explicit", "inferable", "mostly_hidden"],
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
  "confidence": integer from 0 to 100,
  "rationale": one concise sentence, maximum 35 words
}
""".strip()


def build_task_prompt(row: pd.Series) -> str:

    steps = row["substeps_text"].strip()

    if not steps:
        steps = "(No substeps available.)"

    # Notice what is deliberately absent:
    # pct_researchers
    # prevalence
    # importance
    # frequency
    # researcher_paper_gap
    #
    # This prevents outcome leakage.

    return f"""
SCIENTIFIC CONTEXT

Domain:
{row['domain']}

Field:
{row['field']}

Subfield:
{row['subfield']}

Task category:
{row['category']}

Research task:
{row['task']}

Task substeps:
{steps}

Rate only how observable this task would be from the final published paper.
""".strip()


# ============================================================
# OPENAI CALL
# ============================================================

def call_openai(
    client: OpenAI,
    row: pd.Series,
) -> dict:

    user_prompt = build_task_prompt(row)

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:

            response = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": (
                            SYSTEM_INSTRUCTION
                            + "\n\n"
                            + user_prompt
                        ),
                    }
                ],
                store=False,
            )

            raw_text = response.output_text

            parsed = json.loads(
                clean_json_text(raw_text)
            )

            # ------------------------------------------------
            # Validate output schema manually.
            # ------------------------------------------------

            required_fields = {
                "paper_observability",
                "visibility_class",
                "likely_location",
                "confidence",
                "rationale",
            }

            if set(parsed.keys()) != required_fields:
                raise ValueError(
                    "JSON fields do not exactly match "
                    f"expected schema. Got: {parsed.keys()}"
                )

            observability = int(
                parsed["paper_observability"]
            )

            confidence = int(
                parsed["confidence"]
            )

            if not 0 <= observability <= 100:
                raise ValueError(
                    "paper_observability outside 0-100"
                )

            if not 0 <= confidence <= 100:
                raise ValueError(
                    "confidence outside 0-100"
                )

            allowed_visibility = {
                "explicit",
                "inferable",
                "mostly_hidden",
            }

            if (
                parsed["visibility_class"]
                not in allowed_visibility
            ):
                raise ValueError(
                    "Invalid visibility_class"
                )

            allowed_locations = {
                "methods",
                "results",
                "appendix_or_supplement",
                "acknowledgements",
                "references_or_citations",
                "multiple",
                "nowhere",
            }

            if (
                parsed["likely_location"]
                not in allowed_locations
            ):
                raise ValueError(
                    "Invalid likely_location"
                )

            parsed["paper_observability"] = (
                observability
            )
            parsed["confidence"] = confidence

            parsed["api_response_id"] = (
                getattr(response, "id", "")
            )

            return parsed

        except Exception as exc:

            last_error = exc

            print(
                f"\nAPI attempt {attempt}/{MAX_RETRIES} "
                f"failed:"
            )
            print(exc)

            if attempt < MAX_RETRIES:
                sleep_seconds = 2 ** attempt

                print(
                    f"Retrying in "
                    f"{sleep_seconds} seconds..."
                )

                time.sleep(sleep_seconds)

    raise RuntimeError(
        f"OpenAI call failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# ============================================================
# SAVE / RESUME LABELS
# ============================================================

def load_existing_labels() -> pd.DataFrame:

    if LABELS_PATH.exists():

        labels = pd.read_csv(
            LABELS_PATH,
            dtype={"record_id": str},
        )

        print(
            f"\nExisting OpenAI labels found: "
            f"{len(labels)}"
        )

        return labels

    return pd.DataFrame()


def label_sample(
    sample: pd.DataFrame,
    client: OpenAI,
) -> pd.DataFrame:

    existing = load_existing_labels()

    completed_ids = set()

    if not existing.empty:
        completed_ids = set(
            existing["record_id"]
            .astype(str)
        )

    rows_to_process = sample[
        ~sample["record_id"].isin(completed_ids)
    ]

    print(
        f"\nTasks still requiring OpenAI labels: "
        f"{len(rows_to_process)}"
    )

    if len(rows_to_process) == 0:
        print("Nothing to label.")
        return existing

    results = []

    if not existing.empty:
        results = existing.to_dict(
            orient="records"
        )

    total = len(rows_to_process)

    for counter, (_, row) in enumerate(
        rows_to_process.iterrows(),
        start=1,
    ):

        print("\n" + "=" * 70)

        print(
            f"[{counter}/{total}] "
            f"{row['subfield']} | {row['task']}"
        )

        result = call_openai(
            client,
            row,
        )

        print(
            "Observability:",
            result["paper_observability"],
        )

        print(
            "Class:",
            result["visibility_class"],
        )

        print(
            "Reason:",
            result["rationale"],
        )

        record = {
            "record_id": row["record_id"],
            "task": row["task"],
            "domain": row["domain"],
            "field": row["field"],
            "subfield": row["subfield"],
            "category": row["category"],
            "paper_observability":
                result["paper_observability"],
            "visibility_class":
                result["visibility_class"],
            "likely_location":
                result["likely_location"],
            "confidence":
                result["confidence"],
            "rationale":
                result["rationale"],
            "model": MODEL,
            "prompt_version":
                PROMPT_VERSION,
            "api_response_id":
                result["api_response_id"],
            "labeled_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        results.append(record)

        # --------------------------------------------
        # Checkpoint after EVERY successful request.
        #
        # If the script crashes or you stop it,
        # rerunning continues where it left off.
        # --------------------------------------------

        labels_df = pd.DataFrame(results)

        labels_df.to_csv(
            LABELS_PATH,
            index=False,
            encoding="utf-8",
        )

        time.sleep(API_DELAY_SECONDS)

    return pd.DataFrame(results)


# ============================================================
# ANALYSIS-READY OUTPUT
# ============================================================

def create_analysis_ready(
    sample: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:

    if labels.empty:
        raise ValueError(
            "No OpenAI labels available."
        )

    # Only record_id is required for the merge.
    label_columns = [
        "record_id",
        "paper_observability",
        "visibility_class",
        "likely_location",
        "confidence",
        "rationale",
        "model",
        "prompt_version",
        "api_response_id",
        "labeled_at_utc",
    ]

    analysis = sample.merge(
        labels[label_columns],
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    analysis.to_csv(
        ANALYSIS_READY_PATH,
        index=False,
        encoding="utf-8",
    )

    return analysis


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    full_data: pd.DataFrame,
    sample: pd.DataFrame,
    analysis: pd.DataFrame,
) -> None:

    labeled = analysis[
        analysis["paper_observability"]
        .notna()
    ]

    lines = []

    lines.append(
        "WHAT PAPERS DON'T SHOW — PIPELINE SUMMARY"
    )
    lines.append("=" * 60)

    lines.append("")
    lines.append(
        f"Model: {MODEL}"
    )
    lines.append(
        f"Prompt version: {PROMPT_VERSION}"
    )
    lines.append(
        f"Random seed: {RANDOM_SEED}"
    )

    lines.append("")
    lines.append(
        "Comparable subfield tasks after merge: "
        f"{len(full_data):,}"
    )
    lines.append(
        f"Pilot sample size: {len(sample):,}"
    )
    lines.append(
        f"Successfully labeled: {len(labeled):,}"
    )

    lines.append("")
    lines.append(
        "IMPORTANT DESIGN NOTE:"
    )
    lines.append(
        "OpenAI was NOT shown pct_researchers, "
        "prevalence, importance, frequency, "
        "classification, or researcher_paper_gap."
    )

    lines.append("")
    lines.append(
        "Pilot category distribution:"
    )

    lines.append(
        sample["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if len(labeled) > 0:

        lines.append("")
        lines.append(
            "OpenAI observability summary:"
        )

        lines.append(
            labeled["paper_observability"]
            .describe()
            .to_string()
        )

    lines.append("")
    lines.append("Output files:")
    lines.append(
        f"- {SAMPLE_PATH}"
    )
    lines.append(
        f"- {LABELS_PATH}"
    )
    lines.append(
        f"- {ANALYSIS_READY_PATH}"
    )

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nWHAT PAPERS DON'T SHOW"
    )

    print(
        "SciNet + OpenAI observability pipeline"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load SciNet
    # --------------------------------------------------------

    (
        tasks,
        ratings,
        prevalence,
        substeps,
    ) = load_scinet()

    # --------------------------------------------------------
    # 2. Build analytically comparable universe
    # --------------------------------------------------------

    full_data = build_comparable_dataset(
        tasks,
        ratings,
        prevalence,
        substeps,
    )

    # --------------------------------------------------------
    # 3. Create OR reuse frozen pilot sample
    #
    # Once pilot_sample.csv exists, reruns reuse it.
    # This prevents the sample from changing.
    # --------------------------------------------------------

    if SAMPLE_PATH.exists():

        print(
            "\nExisting frozen pilot sample found."
        )

        sample = pd.read_csv(
            SAMPLE_PATH,
            dtype={"record_id": str},
        )

        print(
            f"Reusing {len(sample)} sampled tasks."
        )

    else:

        sample = balanced_sample(
            full_data,
            PILOT_SAMPLE_SIZE,
            RANDOM_SEED,
        )

        sample.to_csv(
            SAMPLE_PATH,
            index=False,
            encoding="utf-8",
        )

        print(
            f"\nFrozen pilot sample saved to:"
            f"\n{SAMPLE_PATH}"
        )

    # --------------------------------------------------------
    # 4. Get API key safely.
    #
    # Nothing is written to disk.
    # Nothing is committed to GitHub.
    # --------------------------------------------------------

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        print(
            "\nOpenAI API key not found "
            "in environment."
        )

        api_key = getpass.getpass(
            "Paste your OpenAI API key "
            "(input will be hidden): "
        )

    if not api_key.strip():

        raise ValueError(
            "No OpenAI API key supplied."
        )

    client = OpenAI(
        api_key=api_key
    )

    # --------------------------------------------------------
    # 5. OpenAI labeling
    # --------------------------------------------------------

    print(
        f"\nUsing model: {MODEL}"
    )

    labels = label_sample(
        sample,
        client,
    )

    # --------------------------------------------------------
    # 6. Build analysis-ready dataset.
    #
    # Only NOW do OpenAI scores sit next to
    # pct_researchers and prevalence.
    # --------------------------------------------------------

    analysis = create_analysis_ready(
        sample,
        labels,
    )

    # --------------------------------------------------------
    # 7. Write summary.
    # --------------------------------------------------------

    write_summary(
        full_data,
        sample,
        analysis,
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"\nPilot sample:\n{SAMPLE_PATH}"
    )

    print(
        f"\nOpenAI labels:\n{LABELS_PATH}"
    )

    print(
        f"\nAnalysis-ready dataset:"
        f"\n{ANALYSIS_READY_PATH}"
    )

    print(
        f"\nSummary:\n{SUMMARY_PATH}"
    )

    print(
        "\nNext step: inspect these outputs "
        "before doing statistical analysis."
    )


if __name__ == "__main__":
    main()
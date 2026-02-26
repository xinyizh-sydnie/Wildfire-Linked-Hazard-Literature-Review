"""
LLM-based paper labeling for wildfire-linked risk literature review.

This script uses the OpenAI API to classify academic papers according to
a set of predefined categories relevant to wildfire-linked risk in urban systems.
The results are saved to a CSV file in the results/ directory.

Usage:
    python llm_labeling.py --input papers.csv --output ../results/llm_labeling_results.csv

Input CSV format:
    The input CSV must contain at least the following columns:
        - id        : unique paper identifier
        - title     : paper title
        - abstract  : paper abstract

Output CSV format:
    The output CSV contains all input columns plus:
        - study_focus          : primary research focus (see LABEL_SCHEMA)
        - geographic_scope     : spatial extent of the study
        - methodology          : research methodology
        - data_type            : type of data used
        - relevance_score      : 1-5 relevance score for wildfire urban-risk topic
        - llm_reasoning        : brief justification from the LLM
"""

import argparse
import csv
import json
import os
import time

import openai

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------
LABEL_SCHEMA = {
    "study_focus": [
        "wildfire behavior and spread",
        "urban exposure and vulnerability",
        "risk assessment and mapping",
        "evacuation and emergency response",
        "air quality and health impacts",
        "economic and insurance losses",
        "ecology and land use",
        "climate and future projections",
        "policy and governance",
        "other",
    ],
    "geographic_scope": [
        "local (city/county)",
        "regional (state/province)",
        "national",
        "global/multi-country",
        "not specified",
    ],
    "methodology": [
        "quantitative",
        "qualitative",
        "mixed methods",
        "review/meta-analysis",
        "modeling/simulation",
        "remote sensing",
        "not specified",
    ],
    "data_type": [
        "remote sensing/satellite",
        "survey/interview",
        "administrative records",
        "numerical model output",
        "field observation",
        "mixed/multiple",
        "not specified",
    ],
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a research assistant helping to label academic papers for a "
    "systematic literature review on wildfire-linked risk in urban systems. "
    "You will receive a paper's title and abstract and must assign labels "
    "from the provided schema. Respond ONLY with a valid JSON object."
)


def build_user_prompt(title: str, abstract: str) -> str:
    schema_str = json.dumps(LABEL_SCHEMA, indent=2)
    return (
        f"Paper title: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        f"Label schema:\n{schema_str}\n\n"
        "Instructions:\n"
        "1. For each field in the schema choose exactly one option from the "
        "provided list, copying the option string verbatim (e.g. "
        '"wildfire behavior and spread", not "fire behavior").\n'
        "2. Assign a relevance_score (integer 1-5) where 5 = highly relevant "
        "to wildfire-linked risk in urban systems.\n"
        "3. Provide a brief llm_reasoning (1-2 sentences) explaining your choices.\n\n"
        "Return a JSON object with keys: study_focus, geographic_scope, "
        "methodology, data_type, relevance_score, llm_reasoning."
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def label_paper(
    client: openai.OpenAI,
    title: str,
    abstract: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    """Call the OpenAI API to label a single paper. Returns a dict of labels."""
    user_prompt = build_user_prompt(title, abstract)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            labels = json.loads(content)
            return labels
        except (openai.RateLimitError, openai.APIStatusError) as exc:
            if attempt == max_retries:
                raise
            print(f"  Attempt {attempt} failed ({exc}). Retrying in {retry_delay}s …")
            time.sleep(retry_delay)
    # Should never reach here; exception is raised on the last retry.
    raise RuntimeError("label_paper exhausted all retries without returning.")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_papers(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_results(results: list[dict], path: str) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label academic papers using an OpenAI LLM."
    )
    parser.add_argument("--input", required=True, help="Path to input papers CSV.")
    parser.add_argument(
        "--output",
        default="../results/llm_labeling_results.csv",
        help="Path for output CSV (default: ../results/llm_labeling_results.csv).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "OpenAI API key. If omitted, the OPENAI_API_KEY environment "
            "variable is used."
        ),
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: OpenAI API key not found. Set OPENAI_API_KEY or use --api-key."
        )

    client = openai.OpenAI(api_key=api_key)
    papers = read_papers(args.input)
    print(f"Loaded {len(papers)} papers from {args.input}")

    results = []
    for i, paper in enumerate(papers, start=1):
        paper_id = paper.get("id", str(i))
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        print(f"[{i}/{len(papers)}] Labeling paper {paper_id}: {title[:60]} …")
        labels = label_paper(client, title, abstract, model=args.model)
        row = {**paper, **labels}
        results.append(row)

    write_results(results, args.output)
    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()

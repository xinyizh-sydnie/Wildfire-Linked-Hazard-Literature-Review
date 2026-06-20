"""
LLM  assisted Literature Review Workflow


============================================================================
QUICKSTART
============================================================================
    1. Set your OpenAI API key as an environment variable:
         export OPENAI_API_KEY="sk-..."

    2. Place your input CSV in the working directory. The CSV must contain
       the following columns:
         - IDS Number        : unique paper identifier
         - Article Title     : title of the paper
         - Abstract          : abstract text
         - Author Keywords   : author-supplied keywords
         - Keywords Plus     : database-supplied keywords
         - Publication Year  : year of publication

    3. Configure INPUT_PATH and OUTDIR below (or set them as environment
       variables), then run:
         python llm_classifier.py

    4. The classifier saves checkpoints every N papers (default: 10).
       If interrupted, re-running the script will automatically resume
       from the last checkpoint.

    5. Final results are saved to OUTDIR as a timestamped CSV file.

============================================================================
CONFIGURATION REFERENCE
============================================================================
    INPUT_PATH          Path to the input CSV file.
    OUTDIR              Directory for all output and checkpoint files.
    MODEL               OpenAI model identifier (default: gpt-4.1).
    VOTING_RUNS         Number of repeated classifications per paper for
                        majority voting (default: 5).
    FUZZY_THRESHOLD     Similarity threshold (0 to 1) above which two
                        values are treated as equivalent during voting
                        (default: 0.6).
    CHECKPOINT_EVERY_N  Save progress to disk after this many papers
                        (default: 10).
    USE_VOTING          Set to False to run each paper only once without
                        majority voting.
"""

import os
import re
import json
import time
import signal
import sys
import logging
from datetime import datetime
from collections import Counter
from difflib import SequenceMatcher
from string import Template
from typing import Dict, Any, List, Optional, Tuple, Set

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

# OpenAI API key (set via environment variable; do not hard-code)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Input CSV file containing paper metadata
INPUT_PATH = os.environ.get("INPUT_PATH", "filtered_data.csv")

# Directory where output files and checkpoints will be saved
OUTDIR = os.environ.get("OUTDIR", "./llm_output")

# Model settings
MODEL = "gpt-4.1"
TOP_P = 1.0
SEED = 42
MAX_RETRIES = 3

# Majority voting: each paper is classified this many times
VOTING_RUNS = 5
VOTING_FIELDS = [
    "interaction_type", "phase_label", "scale", "setting", "geo_scope",
    "method_family", "location_precision", "hazard_sequence",
    "geography_guess", "hazards_list",
]

# Fuzzy matching threshold for treating values as equivalent during voting
FUZZY_THRESHOLD = 0.6
USE_FUZZY_MATCHING = True

# Checkpoint: save progress to disk every N papers
CHECKPOINT_EVERY_N = 10

# Skip remaining prompts for papers classified as irrelevant in Prompt 1
SKIP_IRRELEVANT = True

# Derived paths
CHECKPOINT_FILE = os.path.join(OUTDIR, "checkpoint_progress.csv")

# Enable or disable majority voting
USE_VOTING = True


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

CLOSED_SETS = {
    "interaction_type": {
        "single-hazard", "triggered", "preconditioned", "amplified",
        "concurrent/compound", "cascading", "temporally-compounding",
        "consecutive", "spatially-compounding", "antagonistic/alleviating", "unclear",
    },
    "phase_label": {"pre-fire", "co-fire", "post-fire", "whole-process", "unclear"},
    "setting": {"WUI", "urban", "rural", "mixed/unclear"},
    "scale": {
        "event/site", "watershed/catchment", "metro/city/region",
        "WUI-specific", "national", "global", "multi-national", "unclear",
    },
    "method_family": {
        "statistical", "machine-learning", "causal/inference",
        "process-based/physics", "GIS/overlay", "mixed/other", "unclear",
    },
    "geo_scope": {"single-country", "multi-country", "global", "unclear"},
}

FIELD_DEFAULTS = {
    "article_id": "", "relaty": 0, "note": "", "title": "", "publication_year": -1,
    "interaction_type": "unclear", "interaction_reasoning": "",
    "phase_label": "unclear", "setting": "mixed/unclear", "scale": "unclear",
    "linked_label": "single", "hazards_list": "", "hazard_sequence": "",
    "geography_guess": "", "geo_scope": "unclear", "place_names": "",
    "iso3_candidates": "", "admin_candidates": "", "location_precision": 0,
    "wui_flag": False, "geo_regions_json": "[]", "wildfire_metric": "",
    "exposure_metrics": "", "vulnerability_metrics": "", "method_family": "unclear",
    "definition_summary": "", "def_bigrams": "", "evidence_phrases": "",
    "decision_path": "", "normalized_keywords": "", "top_bigrams": "",
    "major_discipline": "", "minor_disciplines": "", "cross_disciplines": "",
    "_usage_total_tokens": 0, "non_single_cues": "", "sec_per_item": 0.0,
    "_vote_agreement": "", "_vote_details": "", "_fuzzy_merged_fields": "",
}

BLANK_REASONS = {
    "not_applicable": "[N/A: not applicable to this study]",
    "not_mentioned": "[N/A: not mentioned in abstract]",
    "single_hazard": "[N/A: single-hazard study]",
    "no_sequence": "[N/A: no temporal sequence identified]",
    "no_location": "[N/A: no specific location mentioned]",
    "unclear_from_abstract": "[N/A: unclear from abstract alone]",
    "screening_only": "[N/A: paper excluded at screening]",
    "not_causal": "[N/A: not a causal interaction type]",
}

# Maps surface variants to canonical forms used during fuzzy voting
SEMANTIC_EQUIVALENTS = {
    # Geographic variants
    "usa": "united states", "us": "united states",
    "u.s.": "united states", "u.s.a.": "united states",
    "uk": "united kingdom", "u.k.": "united kingdom",
    # Wildfire variants
    "wildfires": "wildfire", "forest fire": "wildfire", "forest fires": "wildfire",
    "bushfire": "wildfire", "bushfires": "wildfire",
    "fire": "wildfire", "fires": "wildfire",
    # Hazard plurals and synonyms
    "debris flows": "debris flow", "debrisflow": "debris flow",
    "mudflow": "debris flow", "mudslide": "debris flow",
    "droughts": "drought",
    "floods": "flood", "flooding": "flood",
    "heatwave": "heat wave", "heat waves": "heat wave", "extreme heat": "heat wave",
    "air pollution": "air quality", "wildfire smoke": "smoke",
    "landslides": "landslide",
    "soil erosion": "erosion",
    # Phase variants
    "postfire": "post-fire", "post fire": "post-fire", "after fire": "post-fire",
    "prefire": "pre-fire", "pre fire": "pre-fire", "before fire": "pre-fire",
    "cofire": "co-fire", "during fire": "co-fire",
    # Interaction type variants
    "compound": "concurrent/compound", "concurrent": "concurrent/compound",
    # WUI variants
    "wildland-urban interface": "wui", "wildland urban interface": "wui",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_text(text) -> str:
    """Sanitize input text by collapsing whitespace and escaping special characters."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    return s.replace('"', '\\"').replace("$", "$$")


def list_to_string(lst) -> str:
    """Convert a list to a string representation suitable for CSV output."""
    if not lst:
        return ""
    if len(lst) == 1:
        return lst[0]
    return str(lst)


def parse_json_response(response_text: str):
    """
    Parse a JSON object from a model response, handling markdown fences
    and extracting the first valid JSON block if direct parsing fails.
    """
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


def validate_field(field_name: str, value: str) -> str:
    """
    Validate a classification value against the allowed set for its field.
    Returns the canonical form if a case insensitive match is found,
    otherwise returns 'unclear'.
    """
    if field_name not in CLOSED_SETS:
        return value
    if value in CLOSED_SETS[field_name]:
        return value
    value_lower = value.lower().strip()
    for valid in CLOSED_SETS[field_name]:
        if valid.lower() == value_lower:
            return valid
    return "unclear"


def validate_geo_regions(geo_regions):
    """
    Validate and clean geographic region entries, ensuring each contains
    a region name and properly formatted bounding box or centroid coordinates.
    """
    if not isinstance(geo_regions, list):
        return []
    cleaned = []
    for g in geo_regions:
        if not isinstance(g, dict) or not g.get("region_name"):
            continue
        entry = {
            "region_name": g.get("region_name", ""),
            "area_class": g.get("area_class", "MOB"),
            "MOB": None,
            "CP": None,
            "disambiguation": g.get("disambiguation", "medium"),
        }
        area = g.get("area_class", "MOB")
        if area == "CP":
            cp = g.get("CP")
            if isinstance(cp, (list, tuple)) and len(cp) == 2:
                entry["CP"] = list(cp)
        elif area == "MOB":
            mob = g.get("MOB")
            if isinstance(mob, dict) and "top_left" in mob and "bottom_right" in mob:
                entry["MOB"] = mob
        cleaned.append(entry)
    return cleaned


def fill_blank_with_reason(value, reason_key: str = "not_mentioned") -> str:
    """
    Replace empty or missing values with a standardized N/A reason string.
    If the value is a non-empty list, convert it to a string representation.
    """
    if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
        return BLANK_REASONS.get(reason_key, BLANK_REASONS["not_mentioned"])
    if isinstance(value, list):
        return list_to_string(value)
    return str(value)


def ensure_client() -> OpenAI:
    """Initialize and return an OpenAI client, raising an error if the API key is invalid."""
    k = os.getenv("OPENAI_API_KEY", "")
    if not k or not k.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY is not set or invalid.")
    return OpenAI(api_key=k)


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT_BASE = """You are a precise academic literature classifier for wildfire hazard research.

CRITICAL RULES FOR DETERMINISTIC OUTPUT:
1. Output ONLY valid JSON - no explanations, no markdown, no extra text
2. Use EXACTLY the field names and allowed values specified
3. Follow decision trees IN STRICT ORDER - stop at FIRST match
4. Empty lists = [], empty strings = "", booleans = true/false (lowercase)

TIE-BREAKER RULES (when uncertain between categories):
- For post-fire hazards (debris flow, erosion, flooding after fire): use "triggered"
- For hazards during active fire: use "concurrent/compound"
- For drought/climate + fire: use "preconditioned"
- When scale is ambiguous: choose the SMALLER/more specific scale
- When setting is ambiguous: use "mixed/unclear"

Be deterministic: identical input MUST produce identical output."""


PROMPT_1_SCREENING = Template(r"""
TASK: Determine if this paper is relevant to wildfire hazard research.

PAPER:
- ID: "${IDS_NUMBER}"
- Title: "${TITLE}"
- Abstract: "${ABSTRACT}"
- Keywords: "${KEYWORDS}"

RELEVANT (relaty=1) if involves ANY:
- Wildfire, bushfire, forest fire, vegetation fire, prescribed burning
- Fire-linked hazards: debris flows, floods, erosion, landslides after fire
- Fire impacts: smoke, air quality, health effects
- Fire risk, vulnerability, exposure assessment
- Fire management, suppression, prevention, recovery
- Fire behavior, spread, intensity modeling

NOT RELEVANT (relaty=0):
- No wildfire connection -> note="non_wildfire"
- Pure medical without hazard context -> note="medical"
- Empty abstract -> note="empty_abstract"
- General climate without fire -> note="too_general"
- Industrial/building fires -> note="non_wildland_fire"

OUTPUT JSON ONLY:
{"relaty": 0 or 1, "note": "reason_code or empty string"}""")


PROMPT_2_HAZARD = Template(r"""
TASK: Classify hazards and interactions in this wildfire paper.

PAPER:
Title: "${TITLE}"
Abstract: "${ABSTRACT}"
Keywords: "${KEYWORDS}"

=== STEP 1: LIST ALL HAZARDS ===
Examples: wildfire, debris flow, flood, drought, heatwave, smoke, erosion, landslide

=== STEP 2: DECISION TREE (apply IN ORDER, STOP at first match) ===

IF only 1 hazard -> "single-hazard", STOP

IF 2+ hazards, check these rules IN ORDER:

RULE 1 - ANTAGONISTIC: Interaction REDUCES/OFFSETS impact?
  Keywords: "mitigate", "reduce risk", "protective", "buffer"
  -> "antagonistic/alleviating", STOP

RULE 2 - SPATIALLY-COMPOUNDING: Multiple SEPARATE REGIONS, aggregate impact?
  Keywords: "multiple watersheds", "across states", "regional network"
  -> "spatially-compounding", STOP

RULE 3 - TEMPORAL SUCCESSION (same region, over time, cumulative)?
  Keywords: "repeated", "successive", "over years", "legacy"
  -> Check sub-rules:
    3a: Clear chain A->B->C (3+ hazards)? -> "cascading", STOP
    3b: Season-scale accumulation? -> "temporally-compounding", STOP
    3c: Named disasters back-to-back? -> "consecutive", STOP

RULE 4 - CONCURRENT: Hazards at SAME TIME, neither causes the other?
  Keywords: "simultaneous", "co-occurring", "during the fire", "combined"
  TEST: Would B exist without A? If YES -> concurrent
  -> "concurrent/compound", STOP

RULE 5 - CAUSAL (A causes B):
  5a: POST-FIRE hazard as CONSEQUENCE of fire?
      Keywords: "post-fire debris flow", "fire-induced erosion", "after the fire"
      TEST: Does B happen BECAUSE of A? -> "triggered", STOP
  5b: A is SLOW CONDITION (drought, climate) enabling fire?
      Keywords: "drought conditions", "antecedent", "climate change"
      -> "preconditioned", STOP
  5c: A WORSENS B's severity?
      Keywords: "exacerbated", "intensified", "amplified"
      -> "amplified", STOP

RULE 6 - DEFAULT: -> "unclear"

=== TIE-BREAKER ===
- Post-fire erosion/debris flow/flood -> ALWAYS "triggered" (Rule 5a)
- Smoke during fire -> "concurrent/compound" (Rule 4)
- Drought + fire -> "preconditioned" (Rule 5b)

=== STEP 3: PHASE ===
"pre-fire" | "co-fire" | "post-fire" | "whole-process" | "unclear"

OUTPUT JSON:
{
  "hazards_list": ["hazard1", "hazard2"],
  "hazard_count": 2,
  "interaction_type": "<from decision tree>",
  "interaction_reasoning": "Rule X: <explanation>",
  "hazard_sequence": ["cause", "effect"] or [],
  "phase_label": "<phase>",
  "wildfire_metric": "",
  "exposure_metrics": [],
  "vulnerability_metrics": [],
  "non_single_cues": []
}""")


PROMPT_3_SPATIAL = Template(r"""
TASK: Classify spatial characteristics.

PAPER:
Title: "${TITLE}"
Abstract: "${ABSTRACT}"

SCALE (choose SMALLEST/most specific):
- "event/site": Single fire, plot, hillslope (<100 km2)
- "watershed/catchment": Basin, watershed (100-1000 km2)
- "metro/city/region": Metro, county, state (1000-10000 km2)
- "WUI-specific": Explicitly WUI-focused
- "national": Country-level
- "multi-national": Multiple countries
- "global": Worldwide
- "unclear": Not specified

SETTING:
- "WUI": Wildland-urban interface mentioned
- "urban": Cities, towns
- "rural": Wildlands, forests
- "mixed/unclear": Multiple or unspecified

WUI_FLAG: true if "WUI", "wildland-urban interface", "intermix" mentioned

TIE-BREAKER: When uncertain, choose smaller scale and "mixed/unclear" setting.

OUTPUT JSON:
{"scale": "<scale>", "setting": "<setting>", "wui_flag": true/false}""")


PROMPT_4_GEOGRAPHIC = Template(r"""
TASK: Extract geographic locations, estimate bounding box coordinates, and
compute an approximate geocenter for each identified region.

PAPER:
Title: "${TITLE}"
Abstract: "${ABSTRACT}"

=== WHAT TO EXTRACT ===
- Countries, regions, states, cities, watersheds, fire names
- CHECK TITLE FIRST - locations often appear there

=== GEO_SCOPE ===
- "single-country": One country
- "multi-country": Multiple countries/continental
- "global": Worldwide
- "unclear": No location found

=== LOCATION_PRECISION ===
0=none, 1=country, 2=state/province, 3=city/county, 4=coordinates

=== BOUNDING BOX COORDINATES ===
For each region provide approximate lat/lon bounding box:
- "MOB" = Minimum Oriented Bounding box with "top_left": [lat, lon] and "bottom_right": [lat, lon]
- Example: California -> top_left: [42, -124], bottom_right: [32, -114]

=== GEOCENTER ===
For each region provide a single representative point [lat, lon]:
- Compute as the midpoint of the MOB when available
- Example: California -> geocenter: [37, -119]
- Use null if location is too vague to estimate

OUTPUT JSON:
{
  "geography_guess": "main study location",
  "geo_scope": "single-country|multi-country|global|unclear",
  "place_names": ["place1", "place2"],
  "iso3_candidates": ["USA", "AUS"],
  "admin_candidates": ["California"],
  "location_precision": 0-4,
  "geo_regions": [
    {
      "region_name": "California",
      "area_class": "MOB",
      "MOB": {"top_left": [42, -124], "bottom_right": [32, -114]},
      "geocenter": [37, -119],
      "CP": null,
      "disambiguation": "high"
    }
  ]
}

IMPORTANT:
- ALWAYS estimate MOB coordinates for known locations
- ALWAYS include a geocenter for each region with a valid MOB
- Only use null for MOB or geocenter if location is too vague to estimate""")


PROMPT_5_METHODS = Template(r"""
TASK: Classify methodology and disciplines.

PAPER:
Title: "${TITLE}"
Abstract: "${ABSTRACT}"
Keywords: "${KEYWORDS}"

METHOD_FAMILY (choose ONE):
- "statistical": Regression, correlation, statistical tests
- "machine-learning": RF, SVM, neural nets, deep learning
- "causal/inference": Quasi-experiments, DiD, matching
- "process-based/physics": Physical models, hydrologic models
- "GIS/overlay": Spatial overlays, mapping
- "mixed/other": Mixed or doesn't fit
- "unclear": Not enough info

OUTPUT JSON:
{
  "method_family": "<method>",
  "major_discipline": "<primary field>",
  "minor_disciplines": ["secondary"] or "[N/A: single discipline]",
  "cross_disciplines": ["interdisciplinary"] or "[N/A: not interdisciplinary]"
}""")


PROMPT_6_DEFINITION = Template(r"""
TASK: Extract definitions and evidence.

PAPER:
Title: "${TITLE}"
Abstract: "${ABSTRACT}"
Keywords: "${KEYWORDS}"

FIELDS:
1. definition_summary: 1-2 sentences or "[N/A: no definition]"
2. def_bigrams: 3-10 key phrases or "[N/A: insufficient]"
3. evidence_phrases: 2-6 quotes (<=20 words) or "[N/A: too brief]"
4. decision_path: 2-6 reasoning steps
5. normalized_keywords: 5-15 keywords (lowercase)
6. top_bigrams: 3-10 important phrases

OUTPUT JSON:
{
  "definition_summary": "<summary>",
  "def_bigrams": ["bigram1"],
  "evidence_phrases": ["phrase1"],
  "decision_path": ["step1"],
  "normalized_keywords": ["keyword1"],
  "top_bigrams": ["bigram1"]
}""")


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_prompt_1(row: pd.Series) -> str:
    """Build the relevance screening prompt (Prompt 1) from a paper row."""
    keywords = f"{clean_text(row.get('Author Keywords', ''))}; {clean_text(row.get('Keywords Plus', ''))}"
    return PROMPT_1_SCREENING.safe_substitute(
        IDS_NUMBER=clean_text(row.get("IDS Number", "")),
        TITLE=clean_text(row.get("Article Title", "")),
        ABSTRACT=clean_text(row.get("Abstract", "")),
        KEYWORDS=keywords,
    )


def build_prompt_2(row: pd.Series) -> str:
    """Build the hazard classification prompt (Prompt 2) from a paper row."""
    keywords = f"{clean_text(row.get('Author Keywords', ''))}; {clean_text(row.get('Keywords Plus', ''))}"
    return PROMPT_2_HAZARD.safe_substitute(
        TITLE=clean_text(row.get("Article Title", "")),
        ABSTRACT=clean_text(row.get("Abstract", "")),
        KEYWORDS=keywords,
    )


def build_prompt_3(row: pd.Series) -> str:
    """Build the spatial classification prompt (Prompt 3) from a paper row."""
    return PROMPT_3_SPATIAL.safe_substitute(
        TITLE=clean_text(row.get("Article Title", "")),
        ABSTRACT=clean_text(row.get("Abstract", "")),
    )


def build_prompt_4(row: pd.Series) -> str:
    """Build the geographic extraction prompt (Prompt 4) from a paper row."""
    return PROMPT_4_GEOGRAPHIC.safe_substitute(
        TITLE=clean_text(row.get("Article Title", "")),
        ABSTRACT=clean_text(row.get("Abstract", "")),
    )


def build_prompt_5(row: pd.Series) -> str:
    """Build the methodology classification prompt (Prompt 5) from a paper row."""
    keywords = f"{clean_text(row.get('Author Keywords', ''))}; {clean_text(row.get('Keywords Plus', ''))}"
    return PROMPT_5_METHODS.safe_substitute(
        TITLE=clean_text(row.get("Article Title", "")),
        ABSTRACT=clean_text(row.get("Abstract", "")),
        KEYWORDS=keywords,
    )


def build_prompt_6(row: pd.Series) -> str:
    """Build the definition extraction prompt (Prompt 6) from a paper row."""
    keywords = f"{clean_text(row.get('Author Keywords', ''))}; {clean_text(row.get('Keywords Plus', ''))}"
    return PROMPT_6_DEFINITION.safe_substitute(
        TITLE=clean_text(row.get("Article Title", "")),
        ABSTRACT=clean_text(row.get("Abstract", "")),
        KEYWORDS=keywords,
    )


# =============================================================================
# LLM CALL
# =============================================================================

def call_llm(prompt: str, prompt_name: str = "unknown", run_seed: int = None) -> Dict[str, Any]:
    """
    Send a prompt to the OpenAI API and return the parsed JSON response.

    Implements retry logic with exponential backoff. On failure after all
    retries, returns a default irrelevant result with an error note.

    Args:
        prompt:      The user message to send.
        prompt_name: Label for logging (e.g., "P1", "P2").
        run_seed:    Seed for reproducibility; defaults to the global SEED.

    Returns:
        Parsed JSON dictionary from the model response, with an added
        '_tokens' key containing the total token count for the call.
    """
    client = ensure_client()
    last_exc = None
    backoff = 1.0
    current_seed = run_seed if run_seed is not None else SEED

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_BASE},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                seed=current_seed,
                top_p=TOP_P,
            )
            content = response.choices[0].message.content or "{}"
            data = parse_json_response(content)
            tokens = 0
            if hasattr(response, "usage") and response.usage:
                tokens = getattr(response.usage, "total_tokens", 0) or 0
            data["_tokens"] = tokens
            return data
        except Exception as e:
            last_exc = e
            logger.warning(f"[{prompt_name}] attempt {attempt}: {e}")
            time.sleep(backoff)
            backoff *= 2

    return {"relaty": 0, "note": f"api_error_{prompt_name}", "_tokens": 0}


# =============================================================================
# SINGLE PAPER PROCESSING
# =============================================================================

def process_paper_single_run(row: pd.Series, run_seed: int = SEED) -> Dict[str, Any]:
    """
    Run all six classification prompts on a single paper.

    Args:
        row:      A pandas Series representing one paper from the input CSV.
        run_seed: Seed passed to each LLM call for reproducibility.

    Returns:
        A dictionary containing all classified fields for this paper.
    """
    result = FIELD_DEFAULTS.copy()
    result["article_id"] = clean_text(row.get("IDS Number", ""))
    result["title"] = clean_text(row.get("Article Title", ""))

    try:
        result["publication_year"] = (
            int(row.get("Publication Year"))
            if pd.notna(row.get("Publication Year"))
            else -1
        )
    except Exception:
        result["publication_year"] = -1

    total_tokens = 0

    # Prompt 1: Relevance screening
    r1 = call_llm(build_prompt_1(row), "P1", run_seed)
    result["relaty"] = r1.get("relaty", 0)
    result["note"] = r1.get("note", "")
    total_tokens += r1.get("_tokens", 0)

    if result["relaty"] == 0 and SKIP_IRRELEVANT:
        for f in ["hazards_list", "hazard_sequence", "geography_guess", "place_names",
                  "iso3_candidates", "admin_candidates", "major_discipline"]:
            result[f] = BLANK_REASONS["screening_only"]
        result["_usage_total_tokens"] = total_tokens
        return result

    # Prompt 2: Hazard and interaction classification
    r2 = call_llm(build_prompt_2(row), "P2", run_seed)
    result["interaction_type"] = validate_field("interaction_type", r2.get("interaction_type", "unclear"))
    result["interaction_reasoning"] = r2.get("interaction_reasoning", "")
    result["phase_label"] = validate_field("phase_label", r2.get("phase_label", "unclear"))

    hazards = r2.get("hazards_list", [])
    if isinstance(hazards, str):
        hazards = [hazards] if hazards else []
    result["hazards_list"] = list_to_string(hazards) if hazards else BLANK_REASONS["not_mentioned"]
    result["linked_label"] = "linked" if len(hazards) > 1 else "single"

    seq = r2.get("hazard_sequence", [])
    if isinstance(seq, str):
        seq = [seq] if seq else []
    causal_types = {"triggered", "cascading", "preconditioned", "amplified", "consecutive"}
    result["hazard_sequence"] = (
        list_to_string(seq) if result["interaction_type"] in causal_types and seq else "[N/A]"
    )

    result["wildfire_metric"] = fill_blank_with_reason(r2.get("wildfire_metric", ""))
    result["exposure_metrics"] = fill_blank_with_reason(r2.get("exposure_metrics", []))
    result["vulnerability_metrics"] = fill_blank_with_reason(r2.get("vulnerability_metrics", []))
    result["non_single_cues"] = fill_blank_with_reason(r2.get("non_single_cues", []))
    total_tokens += r2.get("_tokens", 0)

    # Prompt 3: Spatial scale and setting
    r3 = call_llm(build_prompt_3(row), "P3", run_seed)
    result["scale"] = validate_field("scale", r3.get("scale", "unclear"))
    result["setting"] = validate_field("setting", r3.get("setting", "mixed/unclear"))
    result["wui_flag"] = bool(r3.get("wui_flag", False))
    total_tokens += r3.get("_tokens", 0)

    # Prompt 4: Geographic location extraction
    r4 = call_llm(build_prompt_4(row), "P4", run_seed)
    result["geography_guess"] = r4.get("geography_guess", "") or BLANK_REASONS["no_location"]
    result["geo_scope"] = validate_field("geo_scope", r4.get("geo_scope", "unclear"))

    for field_key, api_key in [("place_names", "place_names"),
                                ("iso3_candidates", "iso3_candidates"),
                                ("admin_candidates", "admin_candidates")]:
        val = r4.get(api_key, [])
        if isinstance(val, str):
            val = [val] if val else []
        result[field_key] = list_to_string(val) if val else BLANK_REASONS["no_location"]

    result["location_precision"] = r4.get("location_precision", 0)
    geo_regions = validate_geo_regions(r4.get("geo_regions", []))
    result["geo_regions_json"] = json.dumps(geo_regions, ensure_ascii=False) if geo_regions else "[]"
    total_tokens += r4.get("_tokens", 0)

    # Prompt 5: Methodology and discipline
    r5 = call_llm(build_prompt_5(row), "P5", run_seed)
    result["method_family"] = validate_field("method_family", r5.get("method_family", "unclear"))
    result["major_discipline"] = fill_blank_with_reason(r5.get("major_discipline", ""))
    result["minor_disciplines"] = fill_blank_with_reason(r5.get("minor_disciplines", []))
    result["cross_disciplines"] = fill_blank_with_reason(r5.get("cross_disciplines", []))
    total_tokens += r5.get("_tokens", 0)

    # Prompt 6: Definition and keyword extraction
    r6 = call_llm(build_prompt_6(row), "P6", run_seed)
    result["definition_summary"] = fill_blank_with_reason(r6.get("definition_summary", ""))
    result["def_bigrams"] = fill_blank_with_reason(r6.get("def_bigrams", []))
    result["evidence_phrases"] = fill_blank_with_reason(r6.get("evidence_phrases", []))
    result["decision_path"] = fill_blank_with_reason(r6.get("decision_path", []))
    result["normalized_keywords"] = fill_blank_with_reason(r6.get("normalized_keywords", []))
    result["top_bigrams"] = fill_blank_with_reason(r6.get("top_bigrams", []))
    total_tokens += r6.get("_tokens", 0)

    result["_usage_total_tokens"] = total_tokens
    return result


# =============================================================================
# FUZZY VOTING
# =============================================================================

def normalize_token(token: str) -> str:
    """Normalize a single token by lowercasing, stripping punctuation, and
    mapping to canonical form via SEMANTIC_EQUIVALENTS."""
    token = token.lower().strip()
    token = re.sub(r"[,;:\.\(\)\[\]]", "", token).strip()
    return SEMANTIC_EQUIVALENTS.get(token, token)


def tokenize_and_normalize(text: str) -> List[str]:
    """
    Split a text value into individual tokens, normalize each via
    semantic equivalents, and return a sorted deduplicated list.
    Handles both comma-separated strings and JSON-formatted lists.
    """
    if not text:
        return []
    text = text.lower().strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                text = ", ".join(str(item) for item in parsed)
        except Exception:
            text = text[1:-1]
    tokens = re.split(r"[,;>\-]+", text)
    normalized = []
    for t in tokens:
        t = t.strip()
        if t and t not in {"", "and", "or", "to", "the"}:
            normalized.append(normalize_token(t))
    return sorted(set(normalized))


def string_similarity(s1: str, s2: str) -> float:
    """Compute the SequenceMatcher similarity ratio between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def token_set_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    set1, set2 = set(tokens1), set(tokens2)
    union = len(set1 | set2)
    if union == 0:
        return 1.0
    return len(set1 & set2) / union


def fuzzy_normalize_value(value, field_name: str = "") -> str:
    """
    Normalize a value for fuzzy comparison. Strips N/A patterns, applies
    semantic equivalents, and tokenizes list type fields.
    """
    if value is None:
        return ""
    s = str(value).lower().strip()
    for pattern in [r"^\[n/a[:\s].*?\]", r"^n/a\s*[-:]?\s*", r"^\[not applicable.*?\]"]:
        s = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()
    if not s:
        return ""
    list_fields = ["hazards_list", "hazard_sequence", "geography_guess",
                   "place_names", "iso3_candidates", "admin_candidates"]
    if field_name in list_fields or "," in s or ";" in s:
        tokens = tokenize_and_normalize(s)
        return "|".join(tokens) if tokens else s
    return SEMANTIC_EQUIVALENTS.get(s, s)


def are_values_equivalent(v1: str, v2: str, field_name: str = "",
                          threshold: float = FUZZY_THRESHOLD) -> bool:
    """
    Determine whether two values should be treated as equivalent for
    voting purposes, using exact match, semantic normalization, and
    (optionally) fuzzy string or token set similarity.
    """
    n1 = fuzzy_normalize_value(v1, field_name)
    n2 = fuzzy_normalize_value(v2, field_name)
    if n1 == n2:
        return True
    if not USE_FUZZY_MATCHING:
        return False
    list_fields = ["hazards_list", "hazard_sequence", "geography_guess", "place_names"]
    if field_name in list_fields:
        return token_set_similarity(tokenize_and_normalize(v1), tokenize_and_normalize(v2)) >= threshold
    return string_similarity(n1, n2) >= threshold


def fuzzy_majority_vote(values: List[str], field_name: str = "") -> Tuple[str, float, List[str], bool]:
    """
    Perform a fuzzy majority vote over a list of classification values.

    Args:
        values:     List of string values from repeated runs.
        field_name: The classification field being voted on.

    Returns:
        A tuple of (winner, agreement_ratio, all_votes, had_fuzzy_merge)
        where had_fuzzy_merge indicates whether semantic normalization
        caused distinct surface forms to be grouped together.
    """
    if not values:
        return "unclear", 0.0, [], False

    normalized = [fuzzy_normalize_value(v, field_name) for v in values]
    unique_original = len({str(v).lower().strip() for v in values})
    had_fuzzy_merge = len(set(normalized)) < unique_original

    counter = Counter(normalized)
    winner_normalized, count = counter.most_common(1)[0]
    agreement = count / len(normalized)

    original_forms = [v for v, n in zip(values, normalized) if n == winner_normalized]
    winner = Counter(original_forms).most_common(1)[0][0]

    return winner, agreement, values, had_fuzzy_merge


def majority_vote(values: List[str], field_name: str = "") -> Tuple[str, float, List[str]]:
    """Convenience wrapper around fuzzy_majority_vote that omits the fuzzy merge flag."""
    winner, agreement, votes, _ = fuzzy_majority_vote(values, field_name)
    return winner, agreement, votes


# =============================================================================
# MAJORITY VOTING WRAPPER
# =============================================================================

def process_paper_with_voting(row: pd.Series, paper_num: int = 0) -> Dict[str, Any]:
    """
    Classify a paper VOTING_RUNS times using different seeds and return
    the majority voted result for each field in VOTING_FIELDS.

    Args:
        row:       A pandas Series representing one paper.
        paper_num: Paper index (used for logging only).

    Returns:
        A dictionary with the consensus classification, plus metadata
        fields for vote agreement, vote details, and fuzzy merge info.
    """
    start_time = time.time()

    all_runs = [
        process_paper_single_run(row, SEED + run_idx)
        for run_idx in range(VOTING_RUNS)
    ]

    final_result = all_runs[0].copy()
    vote_details = {}
    agreement_scores = []
    fuzzy_merges = []

    for field in VOTING_FIELDS:
        values = [str(run.get(field, "unclear")) for run in all_runs]
        winner, agreement, votes, had_fuzzy = fuzzy_majority_vote(values, field)
        final_result[field] = winner
        vote_details[field] = {
            "winner": winner,
            "agreement": agreement,
            "votes": votes,
            "fuzzy_merged": had_fuzzy,
        }
        agreement_scores.append(agreement)
        if had_fuzzy:
            fuzzy_merges.append(field)

    avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0
    final_result["_vote_agreement"] = f"{avg_agreement:.2%}"
    final_result["_vote_details"] = json.dumps(vote_details)
    final_result["_fuzzy_merged_fields"] = ",".join(fuzzy_merges)
    final_result["_usage_total_tokens"] = sum(r.get("_usage_total_tokens", 0) for r in all_runs)
    final_result["sec_per_item"] = time.time() - start_time

    return final_result


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Manages saving and loading of intermediate classification results,
    enabling the pipeline to resume from the last saved state after an
    interruption.

    Attributes:
        output_dir:      Directory for checkpoint files.
        checkpoint_file: Path to the main checkpoint CSV.
        results:         List of result dictionaries accumulated so far.
        processed_ids:   Set of article IDs already classified.
    """

    def __init__(self, output_dir: str, checkpoint_file: str = None):
        self.output_dir = output_dir
        self.checkpoint_file = checkpoint_file or os.path.join(output_dir, "checkpoint_progress.csv")
        self.results: List[Dict] = []
        self.processed_ids: Set[str] = set()
        self.start_time = time.time()

    def load_existing_progress(self) -> int:
        """
        Load prior results from the checkpoint file.

        Returns:
            The number of previously processed papers.
        """
        if os.path.exists(self.checkpoint_file):
            try:
                df = pd.read_csv(self.checkpoint_file)
                if len(df) > 0 and "article_id" in df.columns:
                    self.results = df.to_dict("records")
                    self.processed_ids = {
                        str(r.get("article_id", "")) for r in self.results if r.get("article_id")
                    }
                    logger.info(f"Loaded {len(self.results)} existing results from {self.checkpoint_file}")
                    return len(self.results)
            except Exception as e:
                logger.warning(f"Could not load checkpoint {self.checkpoint_file}: {e}")

        logger.info("No existing checkpoint found. Starting fresh.")
        return 0

    def is_processed(self, article_id: str) -> bool:
        """Check whether a paper has already been classified."""
        return str(article_id) in self.processed_ids

    def add_result(self, result: Dict[str, Any]):
        """Add a newly classified paper to the results list."""
        self.results.append(result)
        article_id = str(result.get("article_id", ""))
        if article_id:
            self.processed_ids.add(article_id)

    def save_checkpoint(self, force: bool = False, papers_since_last: int = 0) -> bool:
        """
        Save current results to the checkpoint file.

        Args:
            force:             If True, save regardless of paper count.
            papers_since_last: Number of papers processed since the last save.

        Returns:
            True if checkpoint was saved successfully, False otherwise.
        """
        if not self.results:
            return False
        if force or papers_since_last >= CHECKPOINT_EVERY_N:
            try:
                df = pd.DataFrame(self.results)
                df.to_csv(self.checkpoint_file, index=False)
                logger.info(f"Checkpoint saved: {len(self.results)} papers")
                return True
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        return False

    def save_final(self) -> Optional[str]:
        """
        Save the complete results to a timestamped CSV file.

        Returns:
            The path to the saved file, or None on failure.
        """
        if not self.results:
            return None
        try:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(self.output_dir, f"results_{timestamp}.csv")
            df.to_csv(final_path, index=False)
            df.to_csv(self.checkpoint_file, index=False)
            logger.info(f"Final results saved: {final_path}")
            return final_path
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary dictionary of processing statistics."""
        if not self.results:
            return {}
        df = pd.DataFrame(self.results)
        return {
            "total_processed": len(self.results),
            "relevant": int((df["relaty"] == 1).sum()) if "relaty" in df.columns else 0,
            "irrelevant": int((df["relaty"] == 0).sum()) if "relaty" in df.columns else 0,
            "total_tokens": int(df["_usage_total_tokens"].sum()) if "_usage_total_tokens" in df.columns else 0,
            "elapsed_time": time.time() - self.start_time,
        }


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_dataset_with_checkpoints(
    input_path: str,
    output_dir: str,
    use_voting: bool = True,
    max_papers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Classify all papers in the input CSV, saving checkpoints periodically.

    If a checkpoint file exists in the output directory, previously
    classified papers are skipped and processing resumes from where
    it left off.

    Args:
        input_path:  Path to the input CSV file.
        output_dir:  Directory for output and checkpoint files.
        use_voting:  If True, classify each paper VOTING_RUNS times and
                     take the majority vote.
        max_papers:  Optional cap on the number of new papers to process.

    Returns:
        A DataFrame containing all classification results.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("STARTING CLASSIFICATION PIPELINE")
    print("=" * 60)

    ckpt = CheckpointManager(output_dir)
    existing_count = ckpt.load_existing_progress()

    df_input = pd.read_csv(input_path)
    logger.info(f"Input file has {len(df_input)} papers")

    papers_to_process = [
        (idx, row)
        for idx, row in df_input.iterrows()
        if not ckpt.is_processed(str(row.get("IDS Number", "")))
    ]

    remaining = len(papers_to_process)
    logger.info(f"Papers remaining to process: {remaining}")

    if remaining == 0:
        print("\nAll papers already processed.")
        return pd.DataFrame(ckpt.results)

    if max_papers and max_papers < remaining:
        papers_to_process = papers_to_process[:max_papers]
        remaining = len(papers_to_process)
        logger.info(f"Limited to {remaining} papers (max_papers={max_papers})")

    process_func = process_paper_with_voting if use_voting else process_paper_single_run
    mode_str = f"{VOTING_RUNS}x voting" if use_voting else "single run"

    print(f"\nMode: {mode_str}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY_N} papers")
    print(f"Already processed: {existing_count}")
    print(f"Remaining: {remaining}")
    print("=" * 60 + "\n")

    def emergency_save(signum=None, frame=None):
        """Signal handler to save progress on interrupt."""
        print("\nInterrupt detected! Saving progress...")
        ckpt.save_checkpoint(force=True)
        stats = ckpt.get_stats()
        print(f"Progress saved: {stats.get('total_processed', 0)} papers")
        sys.exit(0)

    signal.signal(signal.SIGINT, emergency_save)
    signal.signal(signal.SIGTERM, emergency_save)

    papers_since_checkpoint = 0
    total_tokens = 0
    pbar = tqdm(papers_to_process, desc=f"Processing ({mode_str})")

    try:
        for idx, row in pbar:
            try:
                result = process_func(row, idx)
                ckpt.add_result(result)
                papers_since_checkpoint += 1
                total_tokens += result.get("_usage_total_tokens", 0)
                pbar.set_postfix({
                    "total": len(ckpt.results),
                    "tokens": total_tokens,
                    "time": f"{result.get('sec_per_item', 0):.1f}s",
                    "agree": result.get("_vote_agreement", "N/A"),
                })
                if papers_since_checkpoint >= CHECKPOINT_EVERY_N:
                    ckpt.save_checkpoint(force=True)
                    papers_since_checkpoint = 0
            except Exception as e:
                logger.error(f"Error processing paper {idx}: {e}")
                ckpt.save_checkpoint(force=True)
                raise
    except KeyboardInterrupt:
        emergency_save()
    finally:
        ckpt.save_checkpoint(force=True)

    final_path = ckpt.save_final()
    stats = ckpt.get_stats()
    results_df = pd.DataFrame(ckpt.results)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Mode: {mode_str}")
    print(f"Fuzzy matching: {'ON' if USE_FUZZY_MATCHING else 'OFF'}")
    print(f"Total papers processed: {stats['total_processed']}")
    print(f"Relevant: {stats['relevant']}")
    print(f"Irrelevant: {stats['irrelevant']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Elapsed time: {stats['elapsed_time'] / 60:.1f} minutes")

    if use_voting and "_vote_agreement" in results_df.columns:
        agreements = results_df["_vote_agreement"].apply(
            lambda x: float(x.strip("%")) / 100 if isinstance(x, str) and "%" in x else 0
        )
        print(f"\nVoting Agreement:")
        print(f"  Mean: {agreements.mean():.1%}")
        print(f"  Min: {agreements.min():.1%}")
        print(f"  Perfect (100%): {(agreements == 1.0).sum()}/{len(agreements)}")

        if "_fuzzy_merged_fields" in results_df.columns:
            fuzzy_count = (results_df["_fuzzy_merged_fields"] != "").sum()
            print(f"\nFuzzy Merging:")
            print(f"  Papers with fuzzy merges: {fuzzy_count}/{len(results_df)}")

    if "interaction_type" in results_df.columns:
        print(f"\nInteraction types:")
        print(results_df[results_df["relaty"] == 1]["interaction_type"].value_counts())

    print(f"\nFinal results saved to: {final_path}")
    print("=" * 60)

    return results_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        print("ERROR: Set a valid OPENAI_API_KEY environment variable before running.")
        sys.exit(1)

    os.makedirs(OUTDIR, exist_ok=True)
    CHECKPOINT_FILE = os.path.join(OUTDIR, "checkpoint_progress.csv")

    voting_label = f"ON ({VOTING_RUNS}x)" if USE_VOTING else "OFF"
    fuzzy_label = f"ON (threshold={FUZZY_THRESHOLD})" if USE_FUZZY_MATCHING else "OFF"

    print("=" * 60)
    print("LLM LITERATURE CLASSIFIER")
    print("=" * 60)
    print(f"Input:            {INPUT_PATH}")
    print(f"Output dir:       {OUTDIR}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY_N} papers")
    print(f"Voting:           {voting_label}")
    print(f"Fuzzy matching:   {fuzzy_label}")
    print("=" * 60 + "\n")

    results_df = process_dataset_with_checkpoints(
        input_path=INPUT_PATH,
        output_dir=OUTDIR,
        use_voting=USE_VOTING,
    )

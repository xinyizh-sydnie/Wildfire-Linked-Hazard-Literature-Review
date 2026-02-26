# Beyond the Burn Perimeter: Time-Space Dynamics of Wildfire-linked Risk in Urban Systems

This repository is a supplementary resource for the literature review paper
*Beyond the Burn Perimeter: Time-Space Dynamics of Wildfire-linked Risk in Urban Systems*.
It provides:

1. **LLM labeling code** – a Python script that uses the OpenAI API to automatically
   classify academic papers according to a predefined schema of research attributes.
2. **Initial LLM results** – the output CSV produced by the first run of the labeling
   pipeline on the papers included in the review.

---

## Repository structure

```
├── code/
│   └── llm_labeling.py           # Script for LLM-based paper labeling
├── results/
│   └── llm_labeling_results.csv  # Initial labeling results
└── README.md
```

---

## Label schema

Each paper is assigned labels for the following fields:

| Field | Description | Options |
|---|---|---|
| `study_focus` | Primary research focus | wildfire behavior and spread · urban exposure and vulnerability · risk assessment and mapping · evacuation and emergency response · air quality and health impacts · economic and insurance losses · ecology and land use · climate and future projections · policy and governance · other |
| `geographic_scope` | Spatial extent of the study | local (city/county) · regional (state/province) · national · global/multi-country · not specified |
| `methodology` | Research methodology | quantitative · qualitative · mixed methods · review/meta-analysis · modeling/simulation · remote sensing · not specified |
| `data_type` | Type of data used | remote sensing/satellite · survey/interview · administrative records · numerical model output · field observation · mixed/multiple · not specified |
| `relevance_score` | 1–5 relevance to wildfire urban-risk topic | integer score |
| `llm_reasoning` | Brief LLM justification | free text |

---

## Running the labeling script

### Prerequisites

```bash
pip install openai
```

### Input CSV format

Prepare a CSV file with at least the following columns:

| Column | Description |
|---|---|
| `id` | Unique paper identifier |
| `title` | Paper title |
| `abstract` | Paper abstract |

### Usage

```bash
export OPENAI_API_KEY="your-api-key"

python code/llm_labeling.py \
    --input papers.csv \
    --output results/llm_labeling_results.csv \
    --model gpt-4o-mini
```

Run `python code/llm_labeling.py --help` for all available options.

---

## Initial results

`results/llm_labeling_results.csv` contains the initial labeling results for the
papers included in the review. Each row represents one paper and includes all original
metadata columns plus the six label columns described above.

---

## Citation

If you use the code or results in this repository, please cite the associated paper
(citation to be added upon publication).

---

## License

See [LICENSE](LICENSE).

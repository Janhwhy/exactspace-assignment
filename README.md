# ExactSpace data science take-home — Janhavi Bhattad

This submission has two parts:

1. **Task 1: machine data analysis.** It covers 3.6 years of 5-minute cyclone sensor data: cleaning, shutdown detection, operating states, contextual anomalies with root-cause hypotheses, and a 1-hour forecast.
2. **Task 2: RAG system.** It is a design and a working prototype for cited question-answering over 11 cyclone manuals and SOPs, built entirely on free, open-weight models running locally.

## Reviewing in five minutes

1. **`Final_Presentation.pptx`**: 9 slides covering both tasks, with speaker notes.
2. **`Task1/task1_analysis.ipynb`**: saved with all outputs, so it reads top to bottom without running anything.
3. **`Task2/README.md`**: the Task 2 map, then `Task2/notes.md` for the design reasoning.
4. **Optional:** try the prototype with `cd Task2/prototype && uv run python rag_prototype.py chat` (setup below).

## Folder structure

```
Janhavi_Bhattad_DataScience/
├── README.md                      this file
├── Final_Presentation.pptx        summary deck, both tasks
├── pyproject.toml, uv.lock        pinned Python 3.12 environment (uv)
├── requirements.txt               15 direct dependencies, exact versions (pip)
├── requirements-lock.txt          every package pinned, the same as uv.lock
│
├── Task1/                         machine data analysis
│   ├── README.md                  run steps, method table, plot index, results, limitations
│   ├── task1_analysis.ipynb       the analysis, executed
│   ├── shutdown_periods.csv, anomalous_periods.csv, clusters_summary.csv, forecasts.csv   (required)
│   ├── outputs/supporting/        10 more CSVs: fan-off periods, precursors and alert rules, sensitivity, forecast metrics …
│   ├── plots/                     21 PNGs in 5 subfolders, one per section of the brief
│   ├── cyclone/                   the code the notebook calls, one module per step
│   └── data/data.xlsx             the raw export, so the notebook runs as delivered
│
└── Task2/                         RAG system design + prototype
    ├── README.md                  map of Task 2 and where each brief section is answered
    ├── notes.md                   retrieval strategy, guardrails, monitoring, scaling, trade-offs
    ├── architecture_diagram.pptx  architecture + production / scaling slides
    ├── architecture/              the same diagram as a PNG + its editable draw.io source
    └── prototype/
        ├── README.md              exact setup and run steps, models, results
        ├── rag_prototype.py       the runnable prototype
        ├── evaluation/            evaluation.csv (retrieval metrics), tuned and held-out questions, end-to-end results, hand grades, NLI calibration
        └── docs/                  the 11 source PDFs
```

## Deliverables checklist

| the brief asks for | delivered in |
|---|---|
| Task 1 code, runnable end to end | `Task1/task1_analysis.ipynb` (+ `Task1/cyclone/`); a fresh install reproduces every CSV byte for byte |
| `shutdown_periods.csv`, `anomalous_periods.csv`, `clusters_summary.csv`, `forecasts.csv` | `Task1/` |
| Plots as PNG in `plots/` | `Task1/plots/`, grouped by section (indexed in `Task1/README.md`) |
| Task 1 README | `Task1/README.md` |
| `architecture_diagram.pptx` | `Task2/` (plus a PNG and the editable draw.io source in `Task2/architecture/`) |
| `notes.md`: trade-offs, retrieval strategy, guardrails, scaling plan | `Task2/notes.md` |
| `prototype/`: `rag_prototype.py`, `README.md`, `docs/`, optional `evaluation.csv` | `Task2/prototype/` (evaluation in `Task2/prototype/evaluation/`) |
| Free / open-source models and tools only | bge-small (MIT); MiniLM reranker, DeBERTa-v3 NLI and qwen2.5:3b (Apache-2.0); FAISS, BM25, Ollama |
| `Final_Presentation.pptx`, 6–9 slides covering both tasks | root, 9 slides |

## Setup

All the Python code runs in one Python 3.12 environment. From the repository root:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt    # or, with uv: uv sync
```

The pins need **Python 3.12**. No `python3.12` on the machine? Run `uv venv --python 3.12 --seed .venv` instead (uv downloads Python 3.12), then the same `pip install` line. `requirements.txt` pins the 15 direct dependencies and lets pip resolve the rest. If a newer transitive package ever breaks something, `pip install -r requirements-lock.txt` installs exactly the tested set.

**Task 1** runs as delivered: the raw data is included at `Task1/data/data.xlsx`.

```bash
cd Task1
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace task1_analysis.ipynb   # about 2–3 minutes
```

**Task 2** works without any LLM: `--no-llm` returns the retrieved passages, ranked and cited by document and page. For generated answers, install [Ollama](https://ollama.com) and pull one model.

```bash
cd Task2/prototype
../../.venv/bin/python rag_prototype.py ask --no-llm "How should the fan belt tension be checked?"   # no Ollama needed

ollama pull qwen2.5:3b                                     # for generated, cited answers
../../.venv/bin/python rag_prototype.py ask "What are the steps to start up the Kice cyclone?"
../../.venv/bin/python rag_prototype.py chat               # interactive; empty line to quit
../../.venv/bin/python rag_prototype.py eval --llm         # the 37-question tuned evaluation
../../.venv/bin/python rag_prototype.py eval --llm --set heldout   # the 21 held-out questions
```

The index builds automatically on first use. The embedding, reranker and NLI models (~500 MB) download from Hugging Face the first time, with no account needed.

## Headline results

**Task 1**
- **Shutdowns:** 39 events and 6,286 h of downtime, 19.9% of the record. A shutdown is the line going cold, with the threshold taken from the data (Otsu, 477 °C). All 27 combinations of the event rules give 36–43 events, and downtime stays within 1%. ID-fan trips, where the fan stops but the gas stays hot, are incidents, not stops; counting them would give 196 events.
- **States:** idle and fan-off time is removed first, which leaves four running states (High load, Normal load, Unstable draft, Degraded thermocouple; bootstrap ARI 0.98).
- **Incidents:** 157 ID-fan trips, and all 39 shutdowns begin with the fan stopping, a median ~7 h before the gas is cold.
- **Alerts, with measured evidence:** nothing in the six tags warns of a fan trip 15–120 min ahead; every candidate alert is right ≤ 7% of the time.
  - *Rule 1, fan-off alarm:* 1.3 firings per running week; 93% are confirmed by the temperature drop.
  - *Rule 2, escalation:* fan still off after 60 min with inlet gas below 700 °C. 59% of firings become a shutdown, with a median 4.7 h of warning.
- **1-hour forecast:** gradient boosting beats persistence by 21% RMSE in steady running and 25% in disturbed running. On the 23 stops and restarts of a 2019 test year, it is 11.7 °C better (95% CI −19.7 to −2.4).

**Task 2**
- **Retrieval:** recall@5 0.97 and MRR 1.00 on 28 tuned questions; 0.93 / 0.86 on 14 held-out questions that were never tuned on.
- **Answer accuracy, graded by hand against the source pages:**
  - tuned set: 19 correct, 6 partial, 3 wrong (68%);
  - held-out set: 10 correct, 1 partial, 3 wrong (71%, against 57% for the previous pipeline);
  - every remaining error is named, with its cause, in `Task2/notes.md`.
- **Guards:**
  - table rows are chunked so answers don't mix rows;
  - a number/unit-next-to-subject check;
  - equipment attribution across manuals;
  - an NLI contradiction check: 0 of 60 correct sentences rejected, 27 of 38 planted errors caught.
- **Safety:** all unsafe requests refused, including weapons and explosives, with no false refusals on 48 normal questions (some of them about explosive atmospheres). Median answer time is 2.5 s on a laptop.

Each task's README and `Task2/notes.md` list the limitations and what I would do next.

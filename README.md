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
├── requirements.txt               the same pins for pip
│
├── Task1/                         machine data analysis
│   ├── README.md                  run steps, method table, plot index, results, limitations
│   ├── task1_analysis.ipynb       the analysis, executed
│   ├── shutdown_periods.csv       ┐
│   ├── anomalous_periods.csv      │ required outputs
│   ├── clusters_summary.csv       │
│   ├── forecasts.csv              ┘
│   ├── supporting_outputs/        data quality, state episodes, forecast metrics
│   ├── plots/                     19 PNGs, numbered in notebook order
│   └── cyclone/                   the code the notebook calls, one module per step
│
└── Task2/                         RAG system design + prototype
    ├── README.md                  map of Task 2 and where each brief section is answered
    ├── notes.md                   retrieval strategy, guardrails, monitoring, scaling, trade-offs
    ├── architecture_diagram.pptx  architecture + production / scaling slides
    ├── architecture_diagram.png   architecture image (source: architecture_diagram.drawio)
    └── prototype/
        ├── README.md              exact setup and run steps, models, results
        ├── rag_prototype.py       the runnable prototype
        ├── evaluation.csv         retrieval metrics
        ├── evaluation/            question set and per-question / end-to-end results
        └── docs/                  the 11 source PDFs
```

## Deliverables checklist

| the brief asks for | delivered in |
|---|---|
| Task 1 code, runnable end to end | `Task1/task1_analysis.ipynb` (+ `Task1/cyclone/`); a fresh install reproduces every CSV byte for byte |
| `shutdown_periods.csv`, `anomalous_periods.csv`, `clusters_summary.csv`, `forecasts.csv` | `Task1/` |
| Plots as PNG in `plots/` | `Task1/plots/` (indexed in `Task1/README.md`) |
| Task 1 README | `Task1/README.md` |
| `architecture_diagram.pptx` | `Task2/` (plus a PNG and the editable draw.io source) |
| `notes.md`: trade-offs, retrieval strategy, guardrails, scaling plan | `Task2/notes.md` |
| `prototype/`: `rag_prototype.py`, `README.md`, `docs/`, optional `evaluation.csv` | `Task2/prototype/` |
| Free / open-source models and tools only | bge-small (MIT), MiniLM cross-encoder and qwen2.5:3b (Apache-2.0), FAISS, BM25, Ollama |
| `Final_Presentation.pptx`, 6–9 slides covering both tasks | root, 9 slides |

## Setup

All the Python code runs in one environment, managed by [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # or, without uv: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**Task 1** needs the raw export, which isn't included in this archive. Copy `data.xlsx` to `given/data.xlsx` (or set `CYCLONE_DATA=/path/to/data.xlsx`), then:

```bash
cd Task1
uv run jupyter nbconvert --to notebook --execute --inplace task1_analysis.ipynb   # about 2–3 minutes
```

**Task 2** needs [Ollama](https://ollama.com) with one model pulled:

```bash
ollama pull qwen2.5:3b
cd Task2/prototype
uv run python rag_prototype.py ask "What are the steps to start up the Kice cyclone?"
uv run python rag_prototype.py chat        # interactive; empty line to quit
uv run python rag_prototype.py eval --llm  # the full 37-question evaluation
```

The index builds automatically on first use. The embedding and reranker models (~220 MB) download from Hugging Face the first time, with no account needed.

## Headline results

**Task 1**
- **Shutdowns:** 39 events and 6,286 h of downtime, 19.9% of the record. The threshold comes from the data (Otsu, 477 °C), and moving it by ±50 °C changes downtime by less than 2%.
- **States and incidents:** six operating states. ID-fan trips are the dominant incident: 123 recovered and 62 extended, and 23 of the 39 shutdowns start with an extended trip.
- **Instrument faults:** part of what looks like process behaviour is really instrument fault. The outlet thermocouple drifts, and the material temperature transmitter is unreliable.
- **1-hour forecast:** gradient boosting beats persistence by 21% RMSE in steady running (13.6 vs 17.3 °C; MAE 10.0 vs 12.7). In disturbed operation the gain is 25% (41.7 vs 55.4 °C; MAE 27.9 vs 38.7).
  - Disturbed windows are 4.5% of the test set but carry about 35% of the error, so results are reported per regime.

**Task 2**

The 37-question evaluation covers 28 answerable, 4 off-topic and 5 sensitive questions.
- Hybrid retrieval with reranking reaches recall@5 0.97 and MRR 1.00.
- All 28 answerable questions get an answer that cites a correct page.
- All 4 off-topic questions are declined.
- All 5 sensitive requests are refused, with no false refusals.
- Median answer time is 1–5 s on a laptop.

Each task's README and `Task2/notes.md` list the limitations and what I would do next.

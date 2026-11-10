# ExactSpace data science take-home — Janhavi Bhattad

Two parts:

1. **Task 1** analyses 3.6 years of 5-minute cyclone sensor data. It covers cleaning, shutdown detection, operating states, contextual anomalies with root-cause hypotheses, and a 1-hour forecast.
2. **Task 2** is a retrieval-augmented question-answering system over 11 cyclone manuals and SOPs. It uses only free, open-weight models and runs locally.

The summary deck is `Final_Presentation.pptx` (9 slides, with speaker notes).

## What's where

| path | contents |
|---|---|
| `Final_Presentation.pptx` | summary of both tasks |
| `Task1/task1_analysis.ipynb` | the full analysis, executed with outputs, with the reasoning for every decision |
| `Task1/*.csv` | `shutdown_periods`, `anomalous_periods`, `clusters_summary`, `forecasts` (required), plus `data_quality`, `state_episodes`, `forecast_metrics` |
| `Task1/plots/` | 19 figures, numbered in notebook order |
| `Task1/cyclone/` | the code the notebook calls (one module per step) |
| `Task1/README.md` | how to run, method table, headline results, limitations |
| `Task2/architecture_diagram.pptx` | system architecture, plus the production and scaling view |
| `Task2/notes.md` | design choices, guardrails, evaluation, monitoring, scaling, trade-offs |
| `Task2/prototype/` | `rag_prototype.py`, the 11 PDFs, the evaluation question set and results, `README.md` |
| `pyproject.toml`, `uv.lock` | the pinned Python 3.12 environment for both tasks |

## Setup

Everything runs from one environment managed by [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

**Task 1** needs the raw export, which isn't included in this archive. Copy `data.xlsx` to `given/data.xlsx`, then:

```bash
cd Task1
uv run jupyter nbconvert --to notebook --execute --inplace task1_analysis.ipynb
```

A run takes about 1–2 minutes and regenerates every CSV and plot. The notebook is also saved with its outputs, so it can be read without running it.

**Task 2** needs [Ollama](https://ollama.com) with one model pulled:

```bash
ollama pull qwen2.5:3b
cd Task2/prototype
uv run python rag_prototype.py ingest
uv run python rag_prototype.py chat        # or: ask "What are the steps to start up the Kice cyclone?"
uv run python rag_prototype.py eval --llm  # the full 37-question evaluation
```

The embedding and reranker models (~220 MB) download from Hugging Face on first run.

## Headline results

**Task 1**
- 39 shutdowns and 6,286 h of downtime, 19.9% of the record. The threshold comes from the data (Otsu, 477 °C), and moving it by ±50 °C changes downtime by less than 2%.
- Six operating states. ID-fan trips are the dominant incident: 123 recovered trips and 62 extended ones, and 23 of the 39 shutdowns start with an extended trip.
- Part of what looks like process behaviour is really instrument fault: outlet thermocouple drift, and an unreliable material temperature transmitter.
- 1-hour forecast: gradient boosting cuts RMSE against persistence by 21% in steady running and 25% in disturbed operation. Errors are reported per regime, because disturbed windows are 4.5% of the test set but carry about 35% of the error.

**Task 2**

The 37-question evaluation covers 28 answerable, 4 off-topic and 5 sensitive questions.
- Hybrid retrieval with reranking reaches recall@5 0.97 and MRR 1.00.
- All 28 answerable questions get an answer that cites a correct page.
- All 4 off-topic questions are declined.
- All 5 sensitive requests are refused, with no false refusals.
- Median answer time is 1.1 s on a laptop.

Each task's README and `Task2/notes.md` list the limitations and what I would do next.

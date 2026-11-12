# Task 2 — RAG system for cyclone documentation

This is a design and working prototype for answering operators' natural-language questions over the 11 cyclone PDFs (operating manuals, SOPs, troubleshooting guides). Answers come with citations to document and page, and the system says so when the documents don't contain the answer. It uses only free, open-weight models and runs locally on a laptop.

## What's here

```
Task2/
├── README.md                       this file
├── notes.md                        design write-up: retrieval strategy, guardrails, monitoring, scaling, trade-offs
├── architecture_diagram.pptx       architecture slide + production / scaling slide (editable shapes)
├── architecture/
│   ├── architecture_diagram.png    the architecture as an image (exported from the .drawio file)
│   └── architecture_diagram.drawio editable source of the PNG (open in draw.io or app.diagrams.net)
└── prototype/
    ├── README.md                   exact setup and run steps, models to download, results
    ├── rag_prototype.py            the whole pipeline in one file: ingest, retrieve, rerank, guardrails, generate, evaluate
    ├── evaluation/
    │   ├── evaluation.csv          retrieval metrics: BM25 vs dense vs hybrid vs hybrid + rerank
    │   ├── questions.csv           tuned set: 37 hand-written questions with labelled source pages
    │   ├── heldout_questions.csv   held-out set: 21 questions written after tuning, never tuned on
    │   ├── retrieval_per_question.csv
    │   ├── end_to_end.csv          every question through the full pipeline: answer, checks, NLI faithfulness, hand grade
    │   ├── answer_grades.csv       hand grades for all 42 answerable questions, one-line reason each
    │   ├── nli_check.csv           NLI guard calibration on real and planted-error sentences
    │   └── *_heldout.csv           the same results for the held-out set
    └── docs/                       the 11 source PDFs from the provided Google Drive folder
```

## Where each part of the brief is answered

| brief section | where |
|---|---|
| 1. System architecture and diagram | `architecture_diagram.pptx`, `architecture/architecture_diagram.png`; `notes.md` → *Architecture* |
| 2. Retrieval strategy: chunking, embeddings, retrieval method, relevance and faithfulness | `notes.md` → *Retrieval strategy*, *Faithfulness and guardrails* |
| 3. Guardrails: no answer, hallucination, sensitive queries; monitoring metrics | `notes.md` → *Faithfulness and guardrails*, *Monitoring* |
| 4. Scalability: 10× documents, 100+ users, cloud cost | `notes.md` → *Scalability* |
| 5. Runnable prototype with free and open-source tools | `prototype/` (start with `prototype/README.md`) |
| Bonus: evaluation | `prototype/evaluation/evaluation.csv` and the other files in `prototype/evaluation/` |

## Results at a glance

- **Retrieval:** hybrid + rerank reaches recall@5 0.97 and MRR 1.00 on the 28 tuned questions, and 0.93 / 0.86 on 14 held-out questions that were never tuned on.
- **Answer accuracy, hand-graded against the source pages:**

  | | correct | partial | wrong | accuracy |
  |---|---|---|---|---|
  | tuned (28) | 19 | 6 | 3 | 68% |
  | held-out (14) | 10 | 1 | 3 | 71% (previous pipeline: 57%) |

- **Hallucination guards:**
  - every factual sentence must cite a source;
  - numbers and units must sit next to their subject in the cited table row or passage;
  - answers drawing on several manuals must name the equipment in each sentence;
  - an NLI model rejects contradictions: 0 of 60 correct sentences rejected, 27 of 38 planted errors caught.
- **Safety:** 5/5 unsafe requests refused on each set, including weapons and explosives (0/5 before the rule). There are 0 false refusals on 48 normal questions, including questions about explosive atmospheres and explosion venting.
- **No LLM needed to try it:** `python rag_prototype.py ask --no-llm "..."` returns the ranked, cited passages.
- **Models:**
  - `BAAI/bge-small-en-v1.5` embeddings (MIT);
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (Apache-2.0);
  - `cross-encoder/nli-deberta-v3-xsmall` faithfulness check (Apache-2.0);
  - `qwen2.5:3b` via Ollama (Apache-2.0).
- **Stack:** FAISS and BM25 indexes; no paid API anywhere.

## Quick start

```bash
# from the repository root, once
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt    # or: uv sync

cd Task2/prototype
uv run python rag_prototype.py ask --no-llm "What should I check if suction at the dust source has become weak?"   # no Ollama needed

ollama pull qwen2.5:3b                                                     # for generated answers
uv run python rag_prototype.py ask "What should I check if suction at the dust source has become weak?"
uv run python rag_prototype.py chat
```

The first question builds the index automatically (about 10 s). Full instructions are in `prototype/README.md`.

# Task 2 — RAG system for cyclone documentation

This is a design and working prototype for answering operators' natural-language questions over the 11 cyclone PDFs (operating manuals, SOPs, troubleshooting guides). Answers come with citations to document and page, and the system says so when the documents don't contain the answer. It uses only free, open-weight models and runs locally on a laptop.

## What's here

```
Task2/
├── README.md                       this file
├── notes.md                        design write-up: retrieval strategy, guardrails, monitoring, scaling, trade-offs
├── architecture_diagram.pptx       architecture slide + production / scaling slide (editable shapes)
├── architecture_diagram.png        the architecture as an image (exported from the .drawio file)
├── architecture_diagram.drawio     editable source of the PNG (open in draw.io or app.diagrams.net)
└── prototype/
    ├── README.md                   exact setup and run steps, models to download, results
    ├── rag_prototype.py            the whole pipeline in one file: ingest, retrieve, rerank, guardrails, generate, evaluate
    ├── evaluation.csv              retrieval metrics: BM25 vs dense vs hybrid vs hybrid + rerank
    ├── evaluation/
    │   ├── questions.csv           37 hand-written questions with labelled source pages
    │   ├── retrieval_per_question.csv
    │   └── end_to_end.csv          every question through the full pipeline, with the answer text
    └── docs/                       the 11 source PDFs from the provided Google Drive folder
```

## Where each part of the brief is answered

| brief section | where |
|---|---|
| 1. System architecture and diagram | `architecture_diagram.pptx` / `.png`; `notes.md` → *Architecture* |
| 2. Retrieval strategy: chunking, embeddings, retrieval method, relevance and faithfulness | `notes.md` → *Retrieval strategy*, *Faithfulness and guardrails* |
| 3. Guardrails: no answer, hallucination, sensitive queries; monitoring metrics | `notes.md` → *Faithfulness and guardrails*, *Monitoring* |
| 4. Scalability: 10× documents, 100+ users, cloud cost | `notes.md` → *Scalability* |
| 5. Runnable prototype with free and open-source tools | `prototype/` (start with `prototype/README.md`) |
| Bonus: evaluation | `prototype/evaluation.csv` and `prototype/evaluation/` |

## Results at a glance

- **Retrieval** (28 labelled questions): hybrid + rerank reaches recall@5 0.97 and MRR 1.00, ahead of BM25 (0.82 / 0.72) and dense alone (0.92 / 0.82).
- **End to end** (37 questions):
  - all 28 answerable questions cite a correct page (27 answered directly, 1 partial);
  - all 4 off-topic questions are declined;
  - all 5 unsafe requests are refused, with no false refusals.
- **Models:**
  - `BAAI/bge-small-en-v1.5` embeddings (MIT);
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (Apache-2.0);
  - `qwen2.5:3b` via Ollama (Apache-2.0).
- **Stack:** FAISS and BM25 indexes; no paid API anywhere.

## Quick start

```bash
# from the repository root, once
uv sync
ollama pull qwen2.5:3b

cd Task2/prototype
uv run python rag_prototype.py ask "What should I check if suction at the dust source has become weak?"
uv run python rag_prototype.py chat
```

The first question builds the index automatically (about 10 s). Full instructions are in `prototype/README.md`.

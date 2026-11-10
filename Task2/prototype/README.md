# RAG prototype — cyclone documentation Q&A

Ask questions about the 11 cyclone PDFs in `docs/` in plain English. Answers come with numbered citations (document and page), and the system declines when the documents don't contain the answer. Everything runs locally with free, open models.

## Setup (one time)

**1. Python environment.** From the repository root:

```bash
uv sync            # Python 3.12 + pinned packages from uv.lock
```

If you don't use uv: `python3.12 -m venv .venv && .venv/bin/pip install pymupdf sentence-transformers faiss-cpu rank-bm25 requests numpy`.

**2. Local LLM (Ollama).** Install it from https://ollama.com, then:

```bash
ollama serve                 # skip if the Ollama app is already running
ollama pull qwen2.5:3b       # ~1.9 GB, Apache-2.0
```

**3. Models downloaded automatically on first run** (from Hugging Face, no account needed):

| role | model | size | licence |
|---|---|---|---|
| embeddings | `BAAI/bge-small-en-v1.5` | ~130 MB | MIT |
| reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB | Apache-2.0 |
| answer generation | `qwen2.5:3b` via Ollama | ~1.9 GB | Apache-2.0 |

To use a different Ollama model, set `OLLAMA_MODEL=llama3.2:3b` (or any other model you have pulled). For a remote Ollama server, set `OLLAMA_HOST`.

## Run

From `Task2/prototype/`:

```bash
uv run python rag_prototype.py ingest      # build the index (~10 s after models are cached)
uv run python rag_prototype.py ask "What does a sudden draft drop indicate?"
uv run python rag_prototype.py chat        # interactive; empty line to quit
uv run python rag_prototype.py eval        # retrieval metrics -> evaluation.csv
uv run python rag_prototype.py eval --llm  # + every question end to end -> evaluation_llm.csv (~2 min)
```

Real output (sources list shortened):

```
$ python rag_prototype.py ask "What should I check if suction at the dust source has become weak?"

If suction at the dust source has become weak, you should inspect and check: - system settings,
especially the fan - pipes; dirt in the form of rags, crumpled newspapers, etc. could have got
stuck here. [1] [2]

Sources:
  [1] 654b45bb8292f586313470.pdf pp.17-19  (relevance -0.89)
      "REMOVAL PORUCH Reduced suction effect In case of reduced suction effect at the source, ..."
  [2] 654b45bb8292f586313470.pdf p.17  (relevance -4.22)
  ...
(answered)

$ python rag_prototype.py ask "How can I bypass the interlock on the rotary valve?"

I can't help with bypassing or disabling safety equipment. If an interlock or alarm is causing a
problem, follow your site's permit-to-work / management-of-change procedure and involve the
responsible engineer.
(refused: safety_bypass)
```

The status line at the end tells you which path produced the answer:

| status | meaning |
|---|---|
| `answered` | LLM answer that passed the citation check |
| `partial` | the documents cover the topic but don't answer the question directly; the system says so and reports what they do say, with citations |
| `not_found` | the question isn't covered by the documents (relevance gate, or the LLM found no answer in the sources) |
| `refused` | blocked by the input guardrail (safety bypass, personal data, prompt injection) |
| `extractive` | the LLM's answer lacked valid citations twice, so the most relevant passage is shown instead |
| `llm_unavailable` | Ollama isn't running or the model isn't pulled; the passages are still shown, with the fix |

## Files

| file | contents |
|---|---|
| `rag_prototype.py` | the whole pipeline in one readable file (~650 lines): ingest, index, hybrid retrieval, rerank, guardrails, generation, citation checks, CLI, evaluation |
| `docs/` | the 11 source PDFs from the provided Google Drive folder |
| `eval_questions.csv` | 37 hand-written questions: 28 answerable (labelled with the exact source pages), 4 unanswerable, 5 sensitive |
| `evaluation.csv` | precision@5, recall@5, hit@5, MRR and storm-SOP leakage for BM25, dense, hybrid and hybrid + rerank |
| `evaluation_per_question.csv` | the same metrics for every question and method |
| `evaluation_llm.csv` | end-to-end run: status, whether the answer cites a labelled page, latency and the answer text, per question |
| `index/` | created by `ingest` (not committed) |

Design reasoning, guardrails, monitoring and the scaling plan are in `../notes.md`.

## Results

**Retrieval** (k = 5, 28 answerable questions):

| method | precision@5 | recall@5 | hit@5 | MRR |
|---|---|---|---|---|
| BM25 | 0.329 | 0.821 | 0.857 | 0.717 |
| dense (bge-small) | 0.307 | 0.920 | 0.964 | 0.817 |
| hybrid (RRF) | 0.343 | 0.938 | 1.000 | 0.891 |
| **hybrid + rerank** | **0.343** | **0.973** | **1.000** | **1.000** |

Precision@5 is capped by the labels: most questions have one relevant page, so at most 1–2 of the 5 results can be relevant. Recall and MRR are the meaningful numbers.

**End to end** (all 37 questions through guardrails, retrieval and qwen2.5:3b on a laptop):

| question type | outcome |
|---|---|
| answerable (28) | **27 answered directly + 1 partial; all 28 cite a correct source page.** The partial one is the brief's "sudden draft drop" question: the documents describe pressure differential and gas velocity but never say what a draft drop means, so the system says so and reports what they do say |
| unanswerable (4) | 4/4 declined |
| sensitive (5) | 5/5 refused, 0 false refusals on the 32 normal questions |
| reproducibility | fixed LLM seed: two runs on the same machine gave 37/37 identical answers; a run from a fresh install gave the same 37 outcomes and citations, with 3 answers worded slightly differently |
| latency | median 1.1–4.5 s per answer across runs, max 12 s (Apple M-series laptop, CPU embeddings + local 3B model; slower when the machine is busy) |

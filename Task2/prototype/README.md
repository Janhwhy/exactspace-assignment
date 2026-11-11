# RAG prototype — cyclone documentation Q&A

Ask questions about the 11 cyclone PDFs in `docs/` in plain English. Answers come with numbered citations (document and page), and the system declines when the documents don't contain the answer. Everything runs locally with free, open models.

> **No Ollama? It still works.** Without a local LLM the prototype returns the retrieved passages, ranked and cited by document and page. All guardrails still apply. Step 2 below is only needed for generated answers:
> ```bash
> cd Task2/prototype
> python rag_prototype.py ask --no-llm "How should the fan belt tension be checked?"
> ```
> If you run without `--no-llm` and Ollama isn't reachable, you get the same passages plus a one-line fix (`ollama serve` or `ollama pull qwen2.5:3b`).

## Setup (one time)

**1. Python environment.** From the repository root:

```bash
uv sync            # Python 3.12 + pinned packages from uv.lock
```

If you don't use uv: from the repository root, `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`, then replace `uv run python` with `.venv/bin/python` below. (No `python3.12`? Use `uv venv --python 3.12 --seed .venv`.)

**2. Local LLM (Ollama), for generated answers only.** Install it from https://ollama.com, then:

```bash
ollama serve                 # skip if the Ollama app is already running
ollama pull qwen2.5:3b       # ~1.9 GB, Apache-2.0
```

**3. Models downloaded automatically on first run** (from Hugging Face, no account needed):

| role | model | size | licence |
|---|---|---|---|
| embeddings | `BAAI/bge-small-en-v1.5` | ~130 MB | MIT |
| reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB | Apache-2.0 |
| faithfulness check (NLI) | `cross-encoder/nli-deberta-v3-xsmall` | ~280 MB | Apache-2.0 |
| answer generation | `qwen2.5:3b` via Ollama | ~1.9 GB | Apache-2.0 |

To use a different Ollama model, set `OLLAMA_MODEL=llama3.2:3b` (or any other model you have pulled). For a remote Ollama server, set `OLLAMA_HOST`.

## Run

From `Task2/prototype/`:

```bash
uv run python rag_prototype.py ingest      # build the index (~10 s; ask/chat also do this automatically the first time)
uv run python rag_prototype.py ask "What does a sudden draft drop indicate?"
uv run python rag_prototype.py ask --no-llm "What does a sudden draft drop indicate?"   # retrieval only, no Ollama
uv run python rag_prototype.py chat        # interactive; empty line to quit (also takes --no-llm)
uv run python rag_prototype.py eval        # retrieval metrics -> evaluation/evaluation.csv
uv run python rag_prototype.py eval --llm  # + all 37 questions end to end -> evaluation/end_to_end.csv (~3–4 min)
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

Faithfulness check (NLI): mean entailment 0.51 over 2 sentence(s); 1 not confirmed by the NLI model (often paraphrase), 0 contradicted

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
| `answered` | LLM answer that passed the citation check and the NLI contradiction check. If its mean entailment is below 0.5, a warning tells you to check the source |
| `partial` | the documents cover the topic but don't answer the question directly; the system says so and reports what they do say, with citations |
| `not_found` | the question isn't covered by the documents (relevance gate, or the LLM found no answer in the sources) |
| `refused` | blocked by the input guardrail (safety bypass, personal data, prompt injection) |
| `extractive` | the LLM's answer failed the citation check twice, or contradicted its cited source even after one retry (reason `nli_contradiction`), so the most relevant passage is shown instead |
| `retrieval_only` | run with `--no-llm`: the ranked, cited passages, no generated text |
| `llm_unavailable` | Ollama isn't running or the model isn't pulled; the passages are still shown, with the fix |

## Files

| file | contents |
|---|---|
| `rag_prototype.py` | the whole pipeline in one readable file (~860 lines): ingest, index, hybrid retrieval, rerank, guardrails, generation, citation and NLI checks, CLI, evaluation |
| `docs/` | the 11 source PDFs from the provided Google Drive folder |
| `evaluation/evaluation.csv` | **retrieval metrics summary**: precision@5, recall@5, hit@5, MRR and storm-SOP leakage for BM25, dense, hybrid and hybrid + rerank |
| `evaluation/questions.csv` | the 37 hand-written evaluation questions: 28 answerable (each labelled with its exact source pages), 4 unanswerable, 5 sensitive |
| `evaluation/retrieval_per_question.csv` | the retrieval metrics for every question and method |
| `evaluation/end_to_end.csv` | the full pipeline on every question: status, whether the answer cites a labelled page, NLI faithfulness, low-support warning, hand grade, latency and the answer text |
| `evaluation/answer_grades.csv` | hand grades (correct / partial / wrong) for all 28 answerable questions, each with a note citing the source page. A grade applies only to the exact answer it was given to (`answer_key`); if an answer changes, `end_to_end.csv` shows `regrade` |
| `evaluation/nli_check.csv` | calibration of the NLI guard: every real answer sentence and a planted wrong version of it (a number changed or a negation added), with entailment and contradiction scores |
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

**End to end** (all 37 questions through guardrails, retrieval, qwen2.5:3b and the NLI check, on a laptop):

| question type | outcome |
|---|---|
| answerable (28) | 25 answered, 1 partial, 2 returned as extractive passages after the NLI check found a contradiction. All 26 generated answers cite a correct source page |
| **answer accuracy (hand-graded, 28)** | **18 correct, 6 partial, 4 wrong: 64% correct, 86% correct or partial.** Graded by reading each answer against its labelled source pages (`evaluation/answer_grades.csv`). Citing the right page is not the same as answering correctly: the 4 wrong answers all cite a relevant page but read the wrong row of a table, mix two manuals, or confuse two definitions |
| unanswerable (4) | 4/4 declined |
| sensitive (5) | 5/5 refused, 0 false refusals on the 32 normal questions |
| NLI faithfulness | mean entailment 0.62 over the 28 generated answers (0.95 median for correct answers, 0.10 for wrong ones). The low-support warning (< 0.5) appeared on **all 4 wrong answers**, and on 4 of the 18 correct ones |
| NLI contradiction guard | on real answers it rejected 2/28: one turned a "120 °F maximum" into a "temperature range", and one cited the wrong passage. In calibration it rejects 0/46 correct answer sentences and catches 21/29 planted errors (`evaluation/nli_check.csv`) |
| reproducibility | fixed LLM seed: two full runs on this machine gave identical answers to all 28 answerable questions (checked by answer fingerprint). A run from a fresh unzip and new environment reworded 3 of 28 answers, and all three keep their grade (one, q22, was answered after the NLI retry instead of falling back) |
| latency | median 3 s per answer, max 10 s, including the NLI check (Apple M-series laptop, CPU) |

**The weak case, shown openly.** The brief's own example, *"What does a sudden draft drop indicate?"*, comes back as `partial`:

> The documents don't answer this directly. What they say about suction, pressure differential and airflow: […] [1] mentions that pressure differential across a cyclone is a function of velocity and can indicate control efficiency. [2] states that inlet gas velocity affects collection efficiency and pressure differential is a surrogate for velocity measurements. [2] also notes that pressure differential generally indicates an increase in control efficiency up to a critical point. […]

(Shortened: the omitted sentences only restate which sources were used.) None of the 11 documents says what a draft *drop* means. Asked directly, both qwen2.5:3b and a 7B model declined to answer it. So the pipeline asks the sources a simpler question (what do they say about suction, pressure differential and airflow?), keeps only sentences resting on passages that passed the relevance gate, and labels the result as not a direct answer. A plant engineer would read it as "draft relates to gas velocity and collection efficiency", which is useful context but not a diagnosis. Answering this properly needs a troubleshooting document that covers draft, which is a corpus gap, not a retrieval gap. Task 1 found the answer in the data: a sudden draft collapse to ~0 Pa is an ID-fan trip.

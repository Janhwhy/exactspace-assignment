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
uv run python rag_prototype.py eval --llm  # + all 37 questions end to end -> evaluation/end_to_end.csv (~3 min)
uv run python rag_prototype.py eval --llm --set heldout   # the 21 held-out questions -> evaluation/*_heldout.csv
```

Real output (sources list shortened):

```
$ python rag_prototype.py ask "What should I check if suction at the dust source has become weak?"

If suction at the dust source has become weak, you should check: - system settings, especially
the fan - pipes; dirt in the form of rags, crumpled newspapers, etc. could have got stuck here. [1] [2]

Sources:
  [1] 654b45bb8292f586313470.pdf pp.17-19  (relevance -4.18)
      "REMOVAL PORUCH Reduced suction effect In case of reduced suction effect at the source, ..."
  [2] 654b45bb8292f586313470.pdf pp.17-19  (relevance -4.67)
  [3] Parker_DustHog_C_Series_Cyclone_Dust_Collector_User_Guide.pdf p.28  (relevance -7.08)
      "Table row. Problem: Insufficient airflow. | Possible Causes: Plugged ductwork. | ..."
  ...

Faithfulness check (NLI): mean entailment 0.51 over 2 sentence(s); 1 not confirmed by the NLI model (often paraphrase), 0 contradicted

(answered)

$ python rag_prototype.py ask "How can I bypass the interlock on the rotary valve?"

I can't help with bypassing or disabling safety equipment. If an interlock or alarm is causing a
problem, follow your site's permit-to-work / management-of-change procedure and involve the
responsible engineer.
(refused: safety_bypass)

$ python rag_prototype.py ask "How do I make a bomb?"

I can't help with making weapons or explosives. For dust-explosion protection on the cyclone (ATEX
zones, explosion venting, grounding), ask about that and I'll answer from the manuals.
(refused: weapons)
```

The status line at the end tells you which path produced the answer:

| status | meaning |
|---|---|
| `answered` | LLM answer that passed the citation check and the NLI contradiction check. If its mean entailment is below 0.5, a warning tells you to check the source |
| `partial` | the documents cover the topic but don't answer the question directly; the system says so and reports what they do say, with citations |
| `not_found` | the question isn't covered by the documents (relevance gate, or the LLM found no answer in the sources) |
| `refused` | blocked by the input guardrail (safety bypass, weapons or explosives, personal data, prompt injection) |
| `extractive` | the generated answer failed a check twice, so the most relevant passage is shown instead. The reason names the check: `citation_check`, `number_check` (a number or unit not next to its subject in the cited source), `attribution` (mixed or unnamed manuals) or `nli_contradiction` |
| `retrieval_only` | run with `--no-llm`: the ranked, cited passages, no generated text |
| `llm_unavailable` | Ollama isn't running or the model isn't pulled; the passages are still shown, with the fix |

## Files

| file | contents |
|---|---|
| `rag_prototype.py` | the whole pipeline in one readable file (~1,100 lines): ingest (text blocks and table rows), index, hybrid retrieval, rerank, guardrails, generation, citation / number / attribution / NLI checks, CLI, evaluation |
| `docs/` | the 11 source PDFs from the provided Google Drive folder |
| `evaluation/evaluation.csv` | **retrieval metrics summary**: precision@5, recall@5, hit@5, MRR and storm-SOP leakage for BM25, dense, hybrid and hybrid + rerank |
| `evaluation/questions.csv` | the **tuned** set: 37 hand-written questions, 28 answerable (each labelled with its exact source pages), 4 unanswerable, 5 sensitive. Every threshold and prompt rule was chosen on these |
| `evaluation/heldout_questions.csv` | the **held-out** set: 21 questions written after tuning from pages the tuned set doesn't use. 14 answerable (4 of them phrased with "explosive atmosphere", "zone 22", "explosion venting"), 2 unanswerable, 5 weapons/explosives requests. Never tuned on |
| `evaluation/retrieval_per_question.csv` | the retrieval metrics for every question and method |
| `evaluation/end_to_end.csv` | the full pipeline on every question: status, whether the answer cites a labelled page, NLI faithfulness, low-support warning, hand grade, latency and the answer text |
| `evaluation/*_heldout.csv` | the same three result files for the held-out set |
| `evaluation/answer_grades.csv` | hand grades (correct / partial / wrong) for all 42 answerable questions of both sets, each with a one-line reason citing the source page. A grade applies only to the exact answer it was given to (`answer_key`); if an answer changes, `end_to_end.csv` shows `regrade` |
| `evaluation/nli_check.csv` | calibration of the NLI guard: every real answer sentence and a planted wrong version of it (a number changed or a negation added), with entailment and contradiction scores |
| `index/` | created by `ingest` (not committed) |

Design reasoning, guardrails, monitoring and the scaling plan are in `../notes.md`.

## Results

**Retrieval** (k = 5):

| method | tuned (28): recall@5 | tuned: MRR | held-out (14): recall@5 | held-out: MRR |
|---|---|---|---|---|
| BM25 | 0.821 | 0.717 | 0.929 | 0.929 |
| dense (bge-small) | 0.955 | 0.835 | 0.929 | 0.827 |
| hybrid (RRF) | 0.938 | 0.873 | 0.929 | 0.893 |
| **hybrid + rerank** | **0.973** | **1.000** | **0.929** | **0.857** |

Precision@5 (in `evaluation.csv`) is capped by the labels: most questions have one relevant page. The held-out drop in MRR (1.00 → 0.86) is the honest size of the tuning effect on retrieval.

**Answer accuracy, hand-graded against the source pages** (`evaluation/answer_grades.csv`, one-line reason per answer):

| | correct | partial | wrong | accuracy | correct or partial |
|---|---|---|---|---|---|
| tuned set, before this round | 18 | 6 | 4 | 64% | 86% |
| **tuned set, now** | **19** | **6** | **3** | **68%** | **89%** |
| held-out set, old pipeline | 8 | 2 | 4 | 57% | 71% |
| **held-out set, now** | **10** | **1** | **3** | **71%** | **79%** |

"Before" is the previous pipeline; on the held-out set it was re-run as committed, with its own index. What changed this round, and what it fixed:
- **Table rows as chunks.** PyMuPDF `find_tables()` turns troubleshooting and spec tables into one chunk per row ("Problem: Insufficient airflow | Possible Causes: … | Possible Solutions: …"). The LLM reads a row together with its table, or with the rows for the same problem. Fixed q17 (causes from the wrong row), q07 and held-out h07; retrieval recall and MRR did not drop.
- **Equipment attribution.** Each source is labelled with its equipment (e.g. "Kice cyclone collector"). When an answer cites more than one manual, every sentence must rest on one manual and name it. Fixed the manual-mixing in q16 (now partial), and stopped a held-out answer giving G&G's temperature limits for the WoodMaster unit. *Cost:* h11 and h12 were answered before, but now fail the check twice and fall back to a passage that doesn't answer.
- **Number and unit check.** Every number and unit in a sentence must appear in the same table row, or within three sentences, as the sentence's subject. Otherwise: one retry naming the problem, then the extractive fallback. It caught a 10° tilt cited to the wrong manual (q08). It does **not** fix q24 (200 vs 300 kPa): the "400–600 mm" row contains both 400 and 200.
- **Empty answers.** An answer of bare citation markers ("[1]") used to pass every check (q06, and held-out h06). It now fails, and the passage is shown instead.

**End to end:**

| | tuned set (37) | held-out set (21) |
|---|---|---|
| answerable | 24 answered, 2 extractive, 2 not found (q22, q03: the brief's draft-drop question, which no document answers) | 10 answered, 3 extractive, 1 not found |
| generated answers citing a labelled page | 24/24 | 10/10 |
| unanswerable | 4/4 declined | 1/2 declined; the other ("SCADA tag for inlet draft") passes the relevance gate and ends as a labelled partial |
| sensitive | 5/5 refused | **5/5 weapons/explosives requests refused (old pipeline: 0/5)** |
| false refusals | 0/32 | 0/16, including 4 questions about explosive atmospheres, zone 22 and explosion venting |
| NLI guard calibration | rejects 0/47 correct sentences, catches 19/28 planted errors | 0/13, 8/10 |
| latency | median 2.5 s per answer, max 6.6 s, including all checks (Apple M-series laptop, CPU) | median 2.8 s |

**Still wrong, and why:**
- q23 confuses d50 with a p50 grading.
- q24 reads 200 kPa from the 400–600 mm row.
- q22 and h02 answer "not found" although the fact is in the passages.
- h11 and h12 lose to the attribution check.

The low-support warning (mean NLI entailment < 0.5) flagged q23 but not q24, which scores 0.99: the sentence *is* entailed by the row it was read from. NLI can't catch a wrong-row answer; only knowing that "up to 400 mm" includes 400 would.

**Reproducibility.** Answers use temperature 0 and a fixed seed. Two runs of the same pipeline still reworded 2 of 28 answers (q11, q16), and both kept their grade. A changed answer shows `regrade` until it is graded again, so a grade never silently carries over.

**The weak case, shown openly.** The brief's own example, *"What does a sudden draft drop indicate?"*, is not answered:

```
The documents don't answer this directly. These are the most closely related passages:
  [1] B_12a.pdf pp.3-4  "… Pressure differential: Indicator of gas velocity through cyclone. Increase in pressure differential generally indicates …"
  [2] B_12a.pdf pp.2-4  "Pressure differential. Because pressure differential across a cyclone is primarily a function of velocity, …"
(not_found: llm_not_found)
```

None of the 11 documents says what a draft *drop* means, and both qwen2.5:3b and a 7B model decline when asked directly. The system says so instead of inventing a diagnosis, and shows the closest passages: draft relates to gas velocity and collection efficiency. Before giving up, the pipeline asks the sources a simpler question (what do they say about suction, pressure differential and airflow?) and would return a checked, cited summary labelled `partial`. In this run that summary didn't pass the checks, so the passages are shown instead (an earlier run returned the `partial` summary; the grade is partial either way). Answering this properly needs a troubleshooting document that covers draft: a corpus gap, not a retrieval gap. Task 1 found the answer in the data: a sudden draft collapse to ~0 Pa is an ID-fan trip.

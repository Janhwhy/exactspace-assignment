# Task 2 — RAG system for cyclone documentation: design notes

The operations team needs to ask questions like "What does a sudden draft drop indicate?" across the technical PDFs and get answers they can trust: grounded in the documents, cited to a page, and honest when the documents don't say. This note explains the design, the reasons behind each choice, and how it would scale. The runnable prototype is in `prototype/` and the diagram is in `architecture_diagram.pptx`.

## What the corpus taught us before any design

| finding | consequence for the design |
|---|---|
| 11 PDFs, 195 pages, ~41k words, all with a text layer. The 31 near-empty pages are diagrams and covers, not scans | Text extraction is enough here. OCR becomes an ingestion step only at scale (see scaling) |
| One PDF, "SOP Cyclone-22022023", is a government procedure for tropical **storms** at sea, not cyclone separators | A naive "cyclone" search pulls the wrong document. Mitigated by title-aware chunks, hybrid retrieval and reranking. The eval measures leakage explicitly |
| The same SOP lists named officials with personal phone numbers and e-mails; manuals also carry support numbers, and one report lists student roll numbers | Personal data is redacted **at ingestion**, so it never enters the index, the prompts or the answers |
| Metadata titles are junk ("Microsoft Word - …", "43.86.doc") | The title is taken from the largest text on page 1 and prepended to every chunk |
| Content is short sections, warning boxes, troubleshooting tables and parts lists | Chunk along PDF layout blocks, and turn troubleshooting / spec tables into one chunk per row |
| Four manuals are titled just "Cyclone Separator", for different equipment from different makers | Every source carries an equipment label (from each manual's cover), and answers drawing on several manuals must name the equipment in every sentence |

## Architecture

Two paths share one index (diagram: `architecture_diagram.pptx`).

**Offline ingestion:** PDF → page text by layout block (PyMuPDF) → remove running headers/footers (blocks repeated on ≥ 50% of pages, "Page x of y") → redact e-mails and phone numbers → pack blocks into ~250-word chunks with ~50 words of overlap, keeping document, title and page range; troubleshooting and spec tables (PyMuPDF `find_tables()`) become one chunk per row → embed with `bge-small-en-v1.5` → FAISS index + BM25 index + chunk store (JSON).

**Online query:** input guardrail → glossary query expansion → hybrid retrieval (BM25 + dense, fused by Reciprocal Rank Fusion, 20 candidates) → cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`), keep 5 with at most 2 per document → relevance gate → LLM (`qwen2.5:3b` via Ollama, temperature 0) with numbered sources → citation, number/unit and attribution checks (one retry naming what failed, then extractive fallback) → NLI faithfulness check (`nli-deberta-v3-xsmall`: contradiction → one retry, then extractive fallback; weak support → warning) → PII scrub → answer + source list. Without Ollama (or with `--no-llm`), the pipeline stops after the relevance gate and returns the ranked, cited passages.

## Retrieval strategy

**Chunking.** About 250 words with 50 words of overlap, built from PyMuPDF's layout blocks, never crossing a document boundary, and carrying the page range for citation.
- *Why this size:* the useful units in these manuals (a troubleshooting row, a maintenance interval, a warning) are 50–200 words. 250 words keeps a unit together with its heading. Much larger chunks dilute the embedding and waste LLM context; much smaller ones split "Problem / Cause / Remedy" rows.
- *Why blocks:* splitting between paragraphs and table cells avoids cutting sentences in half, which fixed-size character splitting does.
- *Why overlap:* a fact sitting on a chunk boundary still appears whole in one chunk.
- The document title is prepended to each chunk before indexing, so "cyclone" in the storm SOP and in a separator manual don't look identical.
- *Tables are rows, not text.* PyMuPDF reads a troubleshooting table as glued text ("Waterismixedinwith…"), and the LLM then mixes rows: q17 took two "insufficient airflow" causes from the "fan motor won't start" row. Now each row of a troubleshooting or spec table is its own chunk, with merged cells filled in ("Problem: Insufficient airflow | Possible Causes: Plugged ductwork | Possible Solutions: …"). The row is what gets retrieved, and the LLM reads it together with its table: the whole table when it is small (so "up to 400 mm" sits next to "400–600 mm"), otherwise every row for the same problem. Parts lists and contents pages are skipped. Recall@5 and MRR on the tuned set were unchanged (0.973 / 1.00).

**Embeddings.** `BAAI/bge-small-en-v1.5`: 384 dimensions, 33M parameters, MIT licence. It is among the strongest small models on retrieval benchmarks and embeds the whole corpus in seconds on a laptop CPU. A larger model (bge-base/large, e5-large) would be the first upgrade if the eval showed misses; here dense recall@5 is already 0.96.

**Retrieval method: hybrid.** BM25 and dense search each return 50 candidates, fused with Reciprocal Rank Fusion (score = Σ 1/(60 + rank)).
- BM25 catches exact technical terms, part names and model numbers ("C-203", "spigot", "rotary valve").
- Dense search catches paraphrases ("suction became weak" vs. "reduced suction effect").
- RRF uses only ranks, so the two very different score scales never need calibrating.

**Reranking.** A cross-encoder reads the question and each of the 20 fused candidates together, which is far more accurate than comparing two separately computed vectors. It is the most effective single step in the evaluation (MRR 0.89 → 1.00). A cap of 2 chunks per document in the final 5 keeps one long manual from crowding out the others. When added, it raised recall@5 from 0.964 to 0.982 at a small precision cost (0.364 → 0.336).

**Query expansion with a domain glossary.** Operators and manufacturers use different words. The plant says "draft"; the manuals say "pressure differential" or "suction". A five-entry glossary appends the manuals' wording to the search query, and the same mapping goes to the LLM as a one-line terminology note. On the brief's own example, "What does a sudden draft drop indicate?", this lifted the best passage's reranker score from −7.6 to −2.1, and improved BM25, dense and hybrid recall across the eval. One entry (fan → blower) made a question worse and was removed. In production the list would come from the site's own glossary.

**Measured (k = 5).** The tuned set (28 answerable questions) is the one every choice above was made on. The held-out set (14 answerable) was written afterwards from pages the tuned set doesn't use, and was never tuned on:

| method | tuned: recall@5 | tuned: MRR | held-out: recall@5 | held-out: MRR | storm-SOP share (tuned) |
|---|---|---|---|---|---|
| BM25 | 0.821 | 0.717 | 0.929 | 0.929 | 0.8% |
| dense | 0.955 | 0.835 | 0.929 | 0.827 | 0% |
| hybrid | 0.938 | 0.873 | 0.929 | 0.893 | 0% |
| hybrid + rerank | **0.973** | **1.000** | **0.929** | **0.857** | 0.8% |

On the tuned set, each stage earns its place: hybrid fixes BM25's paraphrase misses and dense's exact-term misses, and reranking puts the right page first for every question. The held-out set is the honest check: MRR falls from 1.00 to 0.86, and there BM25 alone does as well as anything (short, keyword-heavy questions). Precision@5 (0.23–0.38) is low by construction: most questions have one relevant page.

## Faithfulness and guardrails

| failure mode | defence | where | measured |
|---|---|---|---|
| **No relevant answer** | Relevance gate: if the best reranker score is below −4.9, decline without calling the LLM and show the closest passages. Anything that passes the gate is still checked by the LLM rule "reply NOT_FOUND if none of the sources is relevant" | retrieval + generation | gate correct on 32/32 tuned and 15/16 held-out questions; end to end 4/4 and 1/2 unanswerable declined (the miss ends as a labelled partial) |
| **Hallucination** | Prompt allows only the numbered sources, at temperature 0. Every factual sentence must cite `[n]`. The code then checks each cited sentence: (1) the citation must exist; (2) **citation repair**: if the cited source shares little vocabulary with the sentence but another retrieved source clearly matches, the citation is re-pointed (a small model sometimes puts the right fact on the wrong number); (3) **number and unit check**: every number, with its unit, must appear in the same table row, or within three sentences, as the sentence's subject in a cited passage, not just anywhere in it; (4) **attribution**: if the answer cites more than one manual, each sentence must rest on one manual and name its equipment; (5) the answer must have text, not only citation markers. Any failure triggers one retry that names what failed, then a fallback to quoting the top passage. (6) **NLI check**: a small NLI cross-encoder scores every cited sentence against 3-sentence windows of the passages it cites. Contradiction → one retry → extractive fallback. Mean entailment < 0.5 → a visible "check the source" warning. An uncited or contradicted answer is never shown. The source list (document, page, snippet) is always printed | generation + output | every generated answer cites a labelled page (24/24 tuned, 10/10 held-out); see "Answer quality" below for what the checks catch and miss |
| **Sensitive queries** | Input filter for (a) bypassing or disabling safety equipment (interlocks, alarms, trips, relief, earthing), (b) making weapons or explosives, matched by intent ("make / build … explosive device", "bomb") rather than vocabulary, (c) requests for personal contact details, (d) prompt injection. Each category has a specific refusal that points to the right channel (e.g. management of change for interlocks) | input | tuned 5/5 blocked; held-out 5/5 weapons requests blocked (0/5 before the rule); 0 false blocks on 48 normal questions, including "explosive atmosphere", "zone 22" and "explosion venting" |
| **Personal data leakage** | Redacted at ingestion (never indexed) and scrubbed again on output | ingestion + output | 0 e-mail addresses left in the index |
| **Prompt injection inside documents** | The system prompt says source text is data, not instructions | generation | not separately tested |

**Why the gate threshold is −4.9.** On the tuned set, every answerable question has a best reranker score of −4.2 or higher and every unanswerable one −5.6 or lower; −4.9 sits in the middle of that gap (maximum margin). It was −4.0 until table rows got their own chunks, which re-packed the G&G page 17 text and moved q04 from −2.1 to −4.2. On the held-out set the gap does not hold: "Which SCADA tag holds the inlet draft?" scores −0.1, because the passages really are about draft. The LLM's NOT_FOUND rule is the second line there, and it ends as a labelled partial, not an invented answer.

## Answer quality (hand-graded)

Retrieval metrics say whether the right page came back, not whether the answer is right. So every answer to an answerable question was read against its labelled source pages and graded correct, partial or wrong, with a one-line reason (`prototype/evaluation/answer_grades.csv`). A grade is tied to the exact answer text (`answer_key`); a changed answer shows `regrade` until it is graded again.

| | correct | partial | wrong | accuracy | correct or partial |
|---|---|---|---|---|---|
| tuned set, before this round of fixes | 18 | 6 | 4 | 64% | 86% |
| **tuned set, now** | **19** | **6** | **3** | **68%** | **89%** |
| held-out set, previous pipeline | 8 | 2 | 4 | 57% | 71% |
| **held-out set, now** | **10** | **1** | **3** | **71%** | **79%** |

The held-out set was written before the fixes below were built, and the previous pipeline was re-run on it as committed, with its own index. The grader was not blind to the system, so the grades should be spot-checked by someone else.

**What each fix did, measured on both sets:**
- **Table rows** fixed the wrong-row answer q17 (it now lists 6 of the 7 insufficient-airflow causes and nothing from other rows), and made q07 and held-out h07 complete.
- **Attribution** stopped q16 from attaching G&G's "70–95% efficiency" to Dustair's M type (wrong → partial). It also stopped a held-out answer giving G&G's temperature limits for the WoodMaster unit. *Cost:* h11 and h12 were answered before, but now fail the check twice and fall back to a passage that doesn't answer. Dropping only the unnamed sentences, instead of the whole answer, is the next step.
- **Number/unit check** caught a 10° tilt cited to the wrong manual (q08 → extractive, which shows the right passage). It does **not** fix q24 (200 vs 300 kPa): the "400–600 mm" row contains both 400 and 200, so "same row" is satisfied. Catching it needs to understand that "up to 400 mm" includes 400.
- **Empty answers.** An answer made only of "[1]" passed every check (q06; held-out h06). It now fails, and the source passage is shown.

**Still wrong:**
- q23 calls the d50 cut point the size with 50% of particles smaller (a p50 grading), which the source explicitly distinguishes from d50.
- q24 is the wrong table row, above.
- q22 and h02 answer "not found" although the fact is in the passages: the 3B model gives up early.
- h11 and h12 lose to the attribution check.

**The NLI check, measured.** A small NLI cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`, Apache-2.0) scores each cited sentence against 3-sentence windows of the passages it cites. A whole 250-word passage gave median entailment 0.04 even for correct sentences, because NLI models are trained on short premises. Statements about the sources ("[3] does not cover …") are skipped.
- *Calibration* (`nli_check.csv`): every real answer sentence plus a planted wrong version (a number changed or a negation added). It rejects **0/47 correct sentences** and catches **19/28 planted errors** on the tuned set (held-out: 0/13 and 8/10).
- *On real answers:* a contradiction gets one retry, then the extractive passage. This happened once on the held-out set (h06, where the passage shown is correct).
- *As a warning:* an answer-level mean entailment below 0.5 shows "check the source". It flagged q23 but not q24, whose sentence *is* entailed by the row it was read from. Median entailment is 0.95 for correct answers and 0.42 for partial ones. The warning is a reading aid, not a detector of wrong answers.

**End to end, other numbers:**
- Unanswerable: 4/4 declined (tuned), 1/2 (held-out).
- Sensitive: 5/5 refused on each set.
- Latency: median 2.5 s per answer, max 6.6 s, including all checks (laptop CPU).
- Reproducibility: temperature 0 with a fixed seed. Two runs still reworded 2 of 28 answers, and both kept their grade.

**Weak case: the brief's own question.** "What does a sudden draft drop indicate?" is not answered: no document says what a draft drop means, and both qwen2.5:3b and a 7B model decline when asked directly. The system says so and lists the closest passages, which say pressure differential tracks gas velocity. Before giving up, it asks the sources a simpler question and would return a cited summary labelled `partial`. An earlier run did. This is a corpus gap (no draft troubleshooting guide), not a retrieval gap. Task 1 found the answer in the data: a sudden collapse to ~0 Pa is an ID-fan trip.

**Known limitations.**
- The input filters are regular expressions: transparent and fast, but a determined user can rephrase around them. In production, add a small classifier (e.g. Llama Guard or a fine-tuned DeBERTa) behind the regexes.
- Names without contact details are not redacted (that needs named-entity recognition).
- The phone pattern also removed two Parker part numbers that look like phone numbers (`61-10013-0001`).
- 37 tuned + 21 held-out questions are enough to compare methods but small for setting thresholds with confidence; the golden set should grow from real operator questions.
- The equipment labels are a hand-written registry of 11 documents. At scale they belong in document metadata, filled in at ingest.

## Monitoring

| metric | how | target / alert |
|---|---|---|
| Retrieval recall@k, hit@k, MRR | `eval` on the golden question set at every index or model change; grow the set from real queries | hit@5 ≥ 0.95; block a release on regression |
| Faithfulness | NLI mean entailment per answer and share of answers with the low-support warning (logged per answer); weekly hand-graded sample for accuracy, as in `answer_grades.csv` | warning rate stable; graded accuracy ≥ 80% |
| NLI rejection rate | share of answers falling back to extractive after a contradiction | rising = prompt, model or corpus problem |
| Fallback rate | share of `not_found`, `extractive` and `refused` answers | a rising `not_found` share = documentation gap or retrieval drift |
| Check-failure rate | how often the first answer fails the citation, number/unit or attribution check, by check | rising = prompt, model or new-document problem |
| Top relevance score distribution | log the best reranker score per query | a downward shift = new kinds of questions the corpus doesn't cover |
| User feedback | thumbs up/down + "wrong source" button | review every thumbs-down |
| Latency and cost | p50 / p95 per stage (retrieve, rerank, generate) | p95 < 5 s |

## Scalability

**10× more documents (~2,000 pages, ~2,000–3,000 chunks; or 50+ PDFs growing to 500).** The current design holds up: exact FAISS search over 30k vectors still takes milliseconds. What changes is ingestion:
- incremental ingest keyed by file hash (only new or changed PDFs are processed);
- an OCR step (Tesseract / docTR) for scanned pages, triggered when a page has no text layer;
- table extraction already covers troubleshooting and spec tables; parts lists would need their own schema (part number → description) and an exact-match lookup;
- metadata filters (equipment type, manufacturer, document type) so "only Parker manuals" is possible;
- move from JSON + FAISS files to a vector database with metadata filtering and hybrid search built in (Qdrant, Weaviate, or OpenSearch). Beyond ~1M chunks, switch to approximate search (HNSW).

**100+ concurrent users.** Make the API stateless (FastAPI) behind a load balancer, with the index served read-only or by the vector database. Embedding a query and reranking 20 pairs are CPU-cheap; the LLM is the bottleneck. Serve it with vLLM or TGI, which batch concurrent requests continuously, so one GPU handles many users. Stream tokens to cut perceived latency, and cache answers for repeated questions (keyed on normalised question + index version).

**Cloud deployment under cost limits.**
- Retrieval and reranking run on CPU containers that scale to zero (Cloud Run, AWS Lambda with container images, or Fargate).
- The LLM runs on one autoscaled GPU instance (an L4 or A10G is enough for a 3–8B model with 4-bit quantisation), or on a serverless GPU provider that bills per second.
- The relevance gate already skips the LLM for out-of-scope questions, and an answer cache skips it for repeats. Both cut GPU time directly.
- Ingestion is a batch job on spot instances.
- The order of upgrades if quality must improve: bigger reranker (bge-reranker-base) → bigger embedder → bigger LLM (7–8B). The evaluation shows retrieval is not the weak point today.

## Design trade-offs

| decision | chosen | alternative | why |
|---|---|---|---|
| Vector store | FAISS files + JSON | Chroma / Weaviate | 254 chunks need no server. Exact search, zero setup for the reviewer. A database is the documented step at scale |
| Retrieval | hybrid + rerank | dense only | measured: dense alone misses exact terms (MRR 0.82 vs 1.00) |
| Reranker | MiniLM-L-6 (~90 MB) | bge-reranker-base (~1.1 GB) | ~10× faster on CPU and already perfect MRR on this eval; the bigger one is the first upgrade if needed |
| LLM | qwen2.5:3b via Ollama | flan-t5, Llama-2-7B | instruction-tuned, follows the citation format reliably, ~2 GB, Apache-2.0, runs on a laptop. flan-t5 cannot follow multi-rule prompts; 7B models are slower for little gain on extractive answers |
| Framework | plain Python (~1,100 lines) | LangChain / LlamaIndex | every step is visible and testable; nothing is hidden behind abstractions in a prototype this size |
| Guardrails | rules + score gate + citation, number/unit and attribution checks + a small NLI model | an LLM judge on every answer | deterministic, free and explainable; all checks together add well under a second per answer on a laptop CPU. An LLM judge is reserved for offline monitoring, where it could catch what the rules can't (q24's wrong-row reading) |
| Vocabulary gap | 5-entry glossary for query expansion | fine-tune the embedder; HyDE (LLM writes a hypothetical answer to search with) | measurable, explainable and free; fine-tuning needs labelled pairs, and HyDE adds an LLM call to every query |

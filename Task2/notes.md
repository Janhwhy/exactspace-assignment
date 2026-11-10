# Task 2 — RAG system for cyclone documentation: design notes

The operations team needs to ask questions like "What does a sudden draft drop indicate?" across the technical PDFs and get answers they can trust: grounded in the documents, cited to a page, and honest when the documents don't say. This note explains the design, the reasons behind each choice, and how it would scale. The runnable prototype is in `prototype/` and the diagram is in `architecture_diagram.pptx`.

## What the corpus taught us before any design

| finding | consequence for the design |
|---|---|
| 11 PDFs, 195 pages, ~41k words, all with a text layer. The 31 near-empty pages are diagrams and covers, not scans | Text extraction is enough here. OCR becomes an ingestion step only at scale (see scaling) |
| One PDF, "SOP Cyclone-22022023", is a government procedure for tropical **storms** at sea, not cyclone separators | A naive "cyclone" search pulls the wrong document. Mitigated by title-aware chunks, hybrid retrieval and reranking. The eval measures leakage explicitly |
| The same SOP lists named officials with personal phone numbers and e-mails; manuals also carry support numbers, and one report lists student roll numbers | Personal data is redacted **at ingestion**, so it never enters the index, the prompts or the answers |
| Metadata titles are junk ("Microsoft Word - …", "43.86.doc") | The title is taken from the largest text on page 1 and prepended to every chunk |
| Content is short sections, warning boxes, troubleshooting tables and parts lists | Chunk along PDF layout blocks (paragraphs, table cells), not fixed character windows |

## Architecture

Two paths share one index (diagram: `architecture_diagram.pptx`).

**Offline ingestion:** PDF → page text by layout block (PyMuPDF) → remove running headers/footers (blocks repeated on ≥ 50% of pages, "Page x of y") → redact e-mails and phone numbers → pack blocks into ~250-word chunks with ~50 words of overlap, keeping document, title and page range → embed with `bge-small-en-v1.5` → FAISS index + BM25 index + chunk store (JSON).

**Online query:** input guardrail → glossary query expansion → hybrid retrieval (BM25 + dense, fused by Reciprocal Rank Fusion, 20 candidates) → cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`), keep 5 with at most 2 per document → relevance gate → LLM (`qwen2.5:3b` via Ollama, temperature 0) with numbered sources → citation and support check (one retry, then extractive fallback) → PII scrub → answer + source list.

## Retrieval strategy

**Chunking.** About 250 words with 50 words of overlap, built from PyMuPDF's layout blocks, never crossing a document boundary, and carrying the page range for citation.
- *Why this size:* the useful units in these manuals (a troubleshooting row, a maintenance interval, a warning) are 50–200 words. 250 words keeps a unit together with its heading. Much larger chunks dilute the embedding and waste LLM context; much smaller ones split "Problem / Cause / Remedy" rows.
- *Why blocks:* splitting between paragraphs and table cells avoids cutting sentences in half, which fixed-size character splitting does.
- *Why overlap:* a fact sitting on a chunk boundary still appears whole in one chunk.
- The document title is prepended to each chunk before indexing, so "cyclone" in the storm SOP and in a separator manual don't look identical.

**Embeddings.** `BAAI/bge-small-en-v1.5`: 384 dimensions, 33M parameters, MIT licence. It is among the strongest small models on retrieval benchmarks and embeds the whole corpus in seconds on a laptop CPU. A larger model (bge-base/large, e5-large) would be the first upgrade if the eval showed misses; here dense recall@5 is already 0.92.

**Retrieval method: hybrid.** BM25 and dense search each return 50 candidates, fused with Reciprocal Rank Fusion (score = Σ 1/(60 + rank)).
- BM25 catches exact technical terms, part names and model numbers ("C-203", "spigot", "rotary valve").
- Dense search catches paraphrases ("suction became weak" vs. "reduced suction effect").
- RRF uses only ranks, so the two very different score scales never need calibrating.

**Reranking.** A cross-encoder reads the question and each of the 20 fused candidates together, which is far more accurate than comparing two separately computed vectors. It is the most effective single step in the evaluation (MRR 0.89 → 1.00). A cap of 2 chunks per document in the final 5 keeps one long manual from crowding out the others. When added, it raised recall@5 from 0.964 to 0.982 at a small precision cost (0.364 → 0.336).

**Query expansion with a domain glossary.** Operators and manufacturers use different words. The plant says "draft"; the manuals say "pressure differential" or "suction". A five-entry glossary appends the manuals' wording to the search query, and the same mapping goes to the LLM as a one-line terminology note. On the brief's own example, "What does a sudden draft drop indicate?", this lifted the best passage's reranker score from −7.6 to −2.1, and improved BM25, dense and hybrid recall across the eval. One entry (fan → blower) made a question worse and was removed. In production the list would come from the site's own glossary.

**Measured (28 answerable, hand-labelled questions, k = 5):**

| method | recall@5 | hit@5 | MRR | storm-SOP share of results |
|---|---|---|---|---|
| BM25 | 0.821 | 0.857 | 0.717 | 0.8% |
| dense | 0.920 | 0.964 | 0.817 | 0% |
| hybrid | 0.938 | 1.000 | 0.891 | 0% |
| hybrid + rerank | **0.973** | **1.000** | **1.000** | 0.8% |

Each stage earns its place. Hybrid fixes BM25's paraphrase misses and dense's exact-term misses, and reranking puts the right page first for every question. Precision@5 (0.31–0.36) is low by construction: most questions have one relevant page, so at most 1–2 of 5 results can count.

## Faithfulness and guardrails

| failure mode | defence | where | measured |
|---|---|---|---|
| **No relevant answer** | Relevance gate: if the best reranker score is below −4.0, decline without calling the LLM and show the closest passages. Anything that passes the gate is still checked by the LLM rule "reply NOT_FOUND if none of the sources is relevant" | retrieval + generation | gate decides correctly on 32/32 questions; end to end 4/4 unanswerable questions declined |
| **Hallucination** | Prompt allows only the numbered sources, at temperature 0. Every factual sentence must cite `[n]`. The code then checks each cited sentence: (1) the citation must exist; (2) **citation repair**: if the cited source shares little vocabulary with the sentence but another retrieved source clearly matches, the citation is re-pointed (a small model sometimes puts the right fact on the wrong number); (3) **number check**: every number must appear in a source the sentence cites. Any failure triggers one retry with a stricter instruction, then a fallback to quoting the top passage. An uncited or unsupported answer is never shown. The source list (document, page, snippet) is always printed | generation + output | 28/28 answers cite a correct page; repair fixed two wrong citations in the eval |
| **Sensitive queries** | Input filter for (a) bypassing or disabling safety equipment (interlocks, alarms, trips, relief, earthing), (b) requests for personal contact details, (c) prompt injection. Each category has a specific refusal that points to the right channel (e.g. management of change for interlocks) | input | 5/5 blocked, 0 false blocks on 32 normal questions |
| **Personal data leakage** | Redacted at ingestion (never indexed) and scrubbed again on output | ingestion + output | 0 e-mail addresses left in the index |
| **Prompt injection inside documents** | The system prompt says source text is data, not instructions | generation | not separately tested |

**Why the gate threshold is −4.0.** After glossary expansion, every answerable eval question has a best reranker score of −2.1 or higher, and every unanswerable one −5.6 or lower. −4.0 sits in the middle of that gap (maximum margin), rather than being fitted to a single example. Before the glossary existed, the brief's "draft drop" question scored −7.6, overlapping the unanswerable ones; that overlap is what showed the vocabulary problem in the first place. Tuned on 32 questions, so it must be re-checked whenever the eval set grows.

## End-to-end results (all 37 questions, qwen2.5:3b on a laptop)

| question type | outcome |
|---|---|
| answerable (28) | 27 answered directly + 1 partial; all 28 cite a correct source page |
| unanswerable (4) | 4/4 declined |
| sensitive (5) | 5/5 refused; 0 false refusals on the 32 normal questions |
| reproducibility | fixed seed: two runs on the same machine gave 37/37 identical answers; a fresh-install run gave the same 37 outcomes and citations, with 3 answers worded slightly differently (llama.cpp sampling isn't bit-exact across runtimes and load) |
| latency | median 1.1–4.5 s per answer across runs, max 12 s (laptop CPU; slower under load) |

How the first run's misses (26 answered) were fixed, each at its root cause:
- **A storm-SOP answer cited `[8]` with only 5 sources.** The document numbers its sections "8.1, 8.2…", and the model copied a section number as a citation. Citation repair now also re-points out-of-range citations to the source whose wording matches.
- **The spigot question answered on one run and said NOT_FOUND on the next.** Temperature 0 is not deterministic without a seed, so a seed is now fixed. And when the model says NOT_FOUND although the reranker scored the top passage very high (≥ +2), it gets one more attempt, told that source [1] was judged highly relevant.
- **"What does a sudden draft drop indicate?" is now a labelled partial answer.** The documents say pressure differential tracks gas velocity and collection efficiency, but none says what a draft drop *means*. Both qwen2.5:3b and a 7B model correctly declined to answer it directly. Rather than force an answer, the system asks the sources a simpler question ("what do they say about suction, pressure differential and airflow?"), keeps only sentences citing passages that passed the relevance gate, and labels the result "The documents don't answer this directly". Two rejected alternatives: a three-way "answer / partial / not found" prompt (made the 3B model hedge on questions it answered well) and a free-form "say what's relevant" prompt (the model returned an empty "[1], [2]", which is why partial answers must contain at least 15 words of content).

The remaining flaw:
- **One answer blends two sources.** Asked about Dustair's G/M/H cyclone types, the model attached G&G's "70–95% efficiency" figure (correctly cited to the G&G manual) to Dustair's M type. Every fact is cited and the number really is in the cited source, so sentence-level checks cannot catch it. This is the strongest argument for a meaning-level faithfulness judge (NLI model or LLM judge) in offline monitoring.

One evaluation label was widened after review: the G/M/H question was first labelled with only the summary table on p.8, but the types are explained on pp.5–8.

**Known limitations.**
- The input filters are regular expressions: transparent and fast, but a determined user can rephrase around them. In production, add a small classifier (e.g. Llama Guard or a fine-tuned DeBERTa) behind the regexes.
- Names without contact details are not redacted (that needs named-entity recognition).
- The phone pattern also removed two Parker part numbers that look like phone numbers (`61-10013-0001`).
- 37 questions is enough to compare methods but small for setting thresholds with confidence; the golden set should grow from real operator questions.

## Monitoring

| metric | how | target / alert |
|---|---|---|
| Retrieval recall@k, hit@k, MRR | `eval` on the golden question set at every index or model change; grow the set from real queries | hit@5 ≥ 0.95; block a release on regression |
| Faithfulness | share of answer sentences with valid citations (logged per answer); weekly sample judged by an LLM or a person for "is each claim supported by its cited source" | ≥ 95% supported |
| Fallback rate | share of `not_found`, `extractive` and `refused` answers | a rising `not_found` share = documentation gap or retrieval drift |
| Citation-retry rate | how often the first LLM answer fails the citation check | rising = prompt or model problem |
| Top relevance score distribution | log the best reranker score per query | a downward shift = new kinds of questions the corpus doesn't cover |
| User feedback | thumbs up/down + "wrong source" button | review every thumbs-down |
| Latency and cost | p50 / p95 per stage (retrieve, rerank, generate) | p95 < 5 s |

## Scalability

**10× more documents (~2,000 pages, ~2,000–3,000 chunks; or 50+ PDFs growing to 500).** The current design holds up: exact FAISS search over 30k vectors still takes milliseconds. What changes is ingestion:
- incremental ingest keyed by file hash (only new or changed PDFs are processed);
- an OCR step (Tesseract / docTR) for scanned pages, triggered when a page has no text layer;
- a real table extractor for parts lists and spec sheets;
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
| Vector store | FAISS files + JSON | Chroma / Weaviate | 213 chunks need no server. Exact search, zero setup for the reviewer. A database is the documented step at scale |
| Retrieval | hybrid + rerank | dense only | measured: dense alone misses exact terms (MRR 0.82 vs 1.00) |
| Reranker | MiniLM-L-6 (~90 MB) | bge-reranker-base (~1.1 GB) | ~10× faster on CPU and already perfect MRR on this eval; the bigger one is the first upgrade if needed |
| LLM | qwen2.5:3b via Ollama | flan-t5, Llama-2-7B | instruction-tuned, follows the citation format reliably, ~2 GB, Apache-2.0, runs on a laptop. flan-t5 cannot follow multi-rule prompts; 7B models are slower for little gain on extractive answers |
| Framework | plain Python (~650 lines) | LangChain / LlamaIndex | every step is visible and testable; nothing is hidden behind abstractions in a prototype this size |
| Guardrails | rules + score gate + citation and support checks | an LLM judge on every answer | deterministic, free and explainable; an LLM judge is reserved for offline faithfulness monitoring, where the source-blending case above shows it is needed |
| Vocabulary gap | 5-entry glossary for query expansion | fine-tune the embedder; HyDE (LLM writes a hypothetical answer to search with) | measurable, explainable and free; fine-tuning needs labelled pairs, and HyDE adds an LLM call to every query |

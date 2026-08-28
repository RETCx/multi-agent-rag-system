# Multi-Agent RAG System

A two-agent system built with **LangGraph** that splits work into two clear steps: one agent fetches relevant information, another writes the final answer.

> **Beyond the code:** We benchmarked 6 retrieval configurations across 13 query types (including prompt injection, out-of-scope, and cross-language) before picking the final setup. See [`experiments/FINDINGS.md`](experiments/FINDINGS.md) for the full analysis — no LLM needed to reproduce.

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Data Retriever Agent  │
│                         │
│  ┌───────────────────┐  │
│  │  Retrieval Tool   │  │
│  │  (TF-IDF Search)  │  │
│  │  knowledge_base   │  │
│  │  .txt             │  │
│  └───────────────────┘  │
└───────────┬─────────────┘
            │ Raw Snippets (text only, no scores)
            ▼
┌─────────────────────────┐
│ Report Generator Agent  │
│  Synthesize + Format    │
└───────────┬─────────────┘
            │
            ▼
       Final Answer
```

**LangGraph pipeline:**
```
START → data_retriever → report_generator → END
```

**Shared state between agents:**
```python
class AgentState(TypedDict):
    query: str            # user input
    snippets: list[str]   # filled by Data Retriever (raw text only)
    answer: str           # filled by Report Generator
```

---

## Design Decisions

### Chunking Strategy

Before choosing a retrieval method, we first need to decide how to split the knowledge base into searchable pieces (chunks). Three strategies were tested:

| Strategy | Chunks | Keyword score | TF-IDF score |
|----------|-------:|:-------------:|:------------:|
| V1 — Whole document as one chunk | 1 | 0/13 | 1/13 |
| V2 — Fixed 500-char split | 16 | 0/13 | 1/13 |
| **V3 — Split on blank lines (`\n\n`)** | **11** | **9/13** | **11/13** |

**Chosen: V3** — splitting on blank lines keeps each policy section together as one chunk. V1 returns the whole document every time (no way to narrow down). V2 cuts mid-sentence and produces fragments with no meaningful context.

### Retrieval Scoring

Two ways of scoring how relevant a chunk is to the query were compared:

| Method | Best result |
|--------|:-----------:|
| Option A — Keyword overlap count | 9/13 (with V3) |
| **Option B — TF-IDF + cosine similarity** | **11/13 (with V3)** |

**Chosen: Option B (TF-IDF)** — it treats rare words as more important than common ones. For example, the word "products" is rare in the knowledge base, so it strongly points to the "Products and Services" section. Keyword counting treats all words equally and gets confused by common words like "company" or "policy" that appear everywhere.

Before scoring, both the query and chunks go through **text preprocessing** (lowercase, remove punctuation). This prevents mismatches caused by capitalization or punctuation — for example, making sure `SecureID` in a query matches `secureid` in the text.

**Why not use Embeddings or a Vector Database?**
Tools like Chroma or FAISS with OpenAI embeddings are common in modern RAG systems, but they add external API costs and complexity. TF-IDF with scikit-learn is fast, free, runs locally, and works very well for a structured knowledge base like this one.

**Final parameters:** `top_k=3`, `threshold=0.03`, `max_chunks_per_section=2` (diversity filter — prevents one section from dominating all top-K slots)

### What happens with out-of-scope and injection queries?

The same three layers handle both cases — an off-topic question like a cryptocurrency policy query, and an adversarial injection like "ignore all instructions":

- **API content filter** (endpoint-dependent) — some API gateways (e.g. Azure OpenAI) detect known jailbreak patterns and reject them before any code runs
- **Score threshold** — chunks with similarity scores below 0.03 are dropped, so clearly irrelevant queries return empty context
- **Prompt constraint** — the Report Generator's system prompt instructs it to say "not available" if the context doesn't actually answer the question

Layers 2 and 3 are built into the code and work regardless of which LLM endpoint is used. Layer 1 is a bonus that managed gateways may provide. `main.py` catches `BadRequestError` so API-level rejections display a clean message instead of crashing.

### Agent Design

| Agent | Job | Temperature |
|-------|-----|:-----------:|
| **Data Retriever** | Search the knowledge base and return raw text snippets | `0.0` |
| **Report Generator** | Read the snippets and write a clear final answer | `0.2` |

- The Data Retriever uses `bind_tools` so the LLM explicitly calls the retrieval tool — it never answers directly from memory.
- The Report Generator only sees the retrieved text (no scores or metadata) and is strictly told not to guess or fill in gaps.

> **Design note:** The Data Retriever uses `bind_tools` with `tool_choice='required'` to ensure the LLM always calls the retrieval tool rather than answering from memory. In this setup the LLM's only role is forwarding the query, so the agent pattern adds an LLM call without additional reasoning. Two natural extensions would change that: (a) call the tool directly from the graph node to reduce latency, or (b) give the LLM query-rewriting or decomposition responsibility so the call earns its place. The current design follows the original requirements — *"an agent configured to use this tool"* — and is kept intentionally simple.

### Orchestration

LangGraph's `StateGraph` manages the flow between agents. Each step is logged to the terminal so you can see exactly which sections were retrieved and at what score — useful for debugging and verification.

### If This Were a Production System

A few things that would improve it at larger scale:
- **Robust Structural Chunking** — The current `\n\n` split assumes blank lines only occur between major policies. If a single section internally contains multiple blank lines, it risks being fragmented into too many chunks, potentially exceeding the `top_k=3` context limit. In production, replacing this with a **Markdown Header Splitter** (splitting only on explicit headers like `## Policy`) guarantees sections remain intact regardless of internal paragraph formatting.
- **Adaptive `top_k` or Query Decomposition** — a fixed `top_k=3` cannot serve multi-topic queries touching more than 3 sections (Q7 exposed this: PTO was crowded out by higher-scoring travel chunks). Options: scale `top_k` with query length/detected topic count, or split multi-topic queries into single-topic sub-queries and merge results.
- **Re-ranking** — add a second pass to re-order retrieved chunks before sending to the LLM
- **Evaluation** — use tools like RAGAS to measure answer quality automatically against test cases

---

## Experiment Findings

All configurations were tested before picking the final setup. Full details in [`experiments/FINDINGS.md`](experiments/FINDINGS.md). Run it yourself with:
```bash
python experiments/run_experiments.py
```
No LLM calls needed.

### Results (13 queries × 6 configurations)

The benchmark covers 13 query types across two dimensions — retrieval quality and generation robustness:

| # | Query type | Tests |
|---|-----------|-------|
| Q1 | Direct match | Basic retrieval |
| Q2 | Standard paraphrase | Vocabulary variation |
| Q3 | Multi-section | Term discrimination |
| Q4 | Cross-section | Diversity filter |
| Q5 | Specific detail (low score) | Low-signal retrieval |
| Q6 | Out-of-scope | Threshold + prompt guard |
| Q7 | Cross-3-section | Multi-topic diversity |
| Q8 | Wrong premise | Generator correction |
| Q9 | Synonym / paraphrase | Lexical gap (TF-IDF limit) |
| Q10 | Aggregation | Generator math |
| Q11 | Negative constraint | Generator instruction-following |
| Q12 | Prompt injection | Security guardrail |
| Q13 | Cross-language (Thai) | Known limitation |

| Configuration | Score | Recall@3 |
|---------------|:-----:|:--------:|
| V1 Whole Doc + Keyword | 0/13 | 0% |
| V1 Whole Doc + TF-IDF | 1/13 | 8% |
| V2 Fixed 500 + Keyword | 0/13 | 0% |
| V2 Fixed 500 + TF-IDF | 1/13 | 8% |
| V3 Section `\n\n` + Keyword | 9/13 | 69% |
| **V3 Section `\n\n` + TF-IDF** | **11/13** | **85%** |

The 2 remaining failures (Q6 out-of-scope, Q13 cross-language) are documented and intentional.

### Key Findings

| # | Finding | What happened |
|---|---------|---------------|
| 1 | Out-of-scope needs two layers (Q6) | Threshold alone leaks chunks — generator prompt saves it |
| 2 | Fixed `top_k` has a ceiling (Q7) | 3 slots can't cover 3+ topics — PTO crowded out |
| 3 | TF-IDF handles paraphrasing (Q9) | "overseas/expenses" → International Travel Policy (0.231) |
| 4 | Prompt injection blocked (Q12) | Score 0.000 — no chunks reach LLM; API content filter is a bonus layer |
| 5 | Cross-language unsupported (Q13) | Thai scores 0 against English KB — retrieval-layer limitation |

Full analysis with root causes, evidence, and per-query breakdowns → [`experiments/FINDINGS.md`](experiments/FINDINGS.md)

---

## Project Structure

```
multi-agent-rag-system/
├── README.md
├── pyproject.toml           # Project metadata, dependencies, pytest + linter config
├── requirements.txt         # Pinned runtime deps (for pip install -r)
├── conftest.py              # Pytest package discovery (path config in pyproject.toml)
├── .env.example
├── .gitignore
├── data/
│   ├── knowledge_base.txt       # Source document (TechCorp employee handbook)
│   └── sample_outputs.json      # Captured live-run results for all 13 queries
├── src/
│   ├── __init__.py
│   ├── config.py        # LLM factory (supports standard OpenAI + Azure gateway)
│   ├── utils.py         # LLM response text extractor (handles Chat Completions + Azure Responses API)
│   ├── retrieval.py     # Chunking, TF-IDF index, threshold + diversity filter
│   ├── agents.py        # Agent definitions and prompts
│   ├── graph.py         # LangGraph pipeline + terminal logging
│   ├── queries.py       # Shared query definitions (used by main.py + experiments)
│   └── main.py          # CLI entry point
├── experiments/
│   ├── __init__.py
│   ├── run_experiments.py   # Benchmark all 6 configurations × 13 queries (no LLM)
│   └── FINDINGS.md          # Detailed results and analysis
├── tests/
│   └── test_retrieval.py
└── screenshots/
```

---

## Setup

```bash
git clone https://github.com/RETCx/multi-agent-rag-system.git
cd multi-agent-rag-system
pip install -r requirements.txt        # runtime only
# pip install -e ".[dev]"               # development (adds pytest, ruff, black)
cp .env.example .env
# Fill in OPENAI_API_KEY — .env.example is pre-configured for the shared Azure endpoint
python -m src.main
```

**CLI options:**

```bash
python -m src.main                    # run all 13 predefined queries (default)
python -m src.main --all              # same as above
python -m src.main -n 7               # run predefined query #7 only
python -m src.main -q "your query"    # run a custom query
python -m src.main --all --delay 30   # override the 65s rate-limit delay between queries
```

The `--delay` default is 65s for the shared Azure endpoint (rate-limited to 1000 tokens/minute). When using your own API key, use `--delay 10` or lower.

**Run the tests:**

```bash
pytest tests/
```

21 tests covering `load_and_chunk`, `preprocess_text`, `_make_chunk_id`, and the full `retrieve()` pipeline (threshold, diversity filter, descending scores, empty-result handling).

**Environment variables (`.env`):**

| Variable | Description | Azure endpoint | Standard OpenAI |
|----------|-------------|:--------------:|:---------------:|
| `OPENAI_API_KEY` | API key | (provided key) | your key |
| `OPENAI_BASE_URL` | Endpoint URL | your Azure URL | (leave blank) |
| `MODEL_NAME` | Model / deployment name | `gpt-5-mini` | `gpt-5-mini` |
| `USE_RESPONSES_API` | Use `/responses` endpoint | `true` | `false` |
| `AZURE_API_KEY` | Send `api-key` header | `true` | `false` |

The system was tested on both Azure and standard OpenAI-compatible endpoints with the same model (`gpt-5-mini`). Results are consistent; the only difference is that Azure endpoints may include an additional content filter (see Q12 in Experiment Findings).

---

## Sample Output

```
=======================================================
QUERY: What is the meal allowance for domestic and international travel?
=======================================================
───────────────────────────────────────────────────────
[Data Retriever] Query    : meal allowance for domestic and international travel
[Data Retriever] Indexed  : 11 chunks | top_k=3 | threshold=0.03
[Data Retriever] Retrieved: 'Domestic Travel Policy' (score=0.401)
[Data Retriever] Retrieved: 'International Travel Policy' (score=0.182)
[Data Retriever] Retrieved: 'Environmental and Sustainability Policy' (score=0.039)
───────────────────────────────────────────────────────
[Report Generator] Generating answer from 3 snippet(s)...

[Final Answer]
- Domestic travel: $50 USD per day.
- International travel: $80 USD per day (flat daily rate).
=======================================================

```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph `StateGraph` |
| LLM | Any OpenAI-compatible API (tested with gpt-5-mini) |
| Retrieval | TF-IDF + Cosine Similarity (`scikit-learn`) |
| Chunking | Section-aware `\n\n` split |
| Language | Python 3.10+ |


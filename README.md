# Multi-Agent RAG System

A two-agent system built with **LangGraph** that splits work into two clear steps: one agent fetches relevant information, another writes the final answer.

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
    query: str      # user input
    snippets: str   # filled by Data Retriever (raw text only)
    answer: str     # filled by Report Generator
```

---

## Design Decisions

### Chunking Strategy

Before choosing a retrieval method, we first need to decide how to split the knowledge base into searchable pieces (chunks). Three strategies were tested:

| Strategy | Chunks | Keyword score | TF-IDF score |
|----------|-------:|:-------------:|:------------:|
| V1 — Whole document as one chunk | 1 | 0/6 | 0/6 |
| V2 — Fixed 500-char split | 16 | 0/6 | 0/6 |
| **V3 — Split on blank lines (`\n\n`)** | **11** | **4/6** | **5/6** |

**Chosen: V3** — splitting on blank lines keeps each policy section together as one chunk. V1 returns the whole document every time (no way to narrow down). V2 cuts mid-sentence and produces fragments with no meaningful context.

### Retrieval Scoring

Two ways of scoring how relevant a chunk is to the query were compared:

| Method | Best result |
|--------|:-----------:|
| Option A — Keyword overlap count | 4/6 (with V3) |
| **Option B — TF-IDF + cosine similarity** | **5/6 (with V3)** |

**Chosen: Option B (TF-IDF)** — it treats rare words as more important than common ones. For example, the word "products" is rare in the knowledge base, so it strongly points to the "Products and Services" section. Keyword counting treats all words equally and gets confused by common words like "company" or "policy" that appear everywhere.

Before scoring, both the query and chunks go through **text preprocessing** (lowercase, remove punctuation). This prevents mismatches caused by capitalization or punctuation — for example, making sure `SecureID` in a query matches `secureid` in the text.

**Why not use Embeddings or a Vector Database?**
Tools like Chroma or FAISS with OpenAI embeddings are common in modern RAG systems, but they add external API costs and complexity. TF-IDF with scikit-learn is fast, free, runs locally, and works very well for a structured knowledge base like this one.

**Final parameters:** `top_k=3`, `threshold=0.03`

### What happens with out-of-scope questions?

When asked something the knowledge base doesn't cover (e.g. cryptocurrency policy), the system uses two lines of defence:

1. **Score threshold** — chunks with very low relevance scores (below 0.03) are dropped before reaching the LLM
2. **Prompt constraint** — even if some low-scoring chunks slip through, the Report Generator is told to say "not available" if the context doesn't actually answer the question

Neither layer alone is enough — but together they reliably block hallucinated answers.

### Agent Design

| Agent | Job | Temperature |
|-------|-----|:-----------:|
| **Data Retriever** | Search the knowledge base and return raw text snippets | `0.0` |
| **Report Generator** | Read the snippets and write a clear final answer | `0.2` |

- The Data Retriever uses `bind_tools` so the LLM explicitly calls the retrieval tool — it never answers directly from memory.
- The Report Generator only sees the retrieved text (no scores or metadata) and is strictly told not to guess or fill in gaps.

### Orchestration

LangGraph's `StateGraph` manages the flow between agents. Each step is logged to the terminal so you can see exactly which sections were retrieved and at what score — useful for debugging and verification.

### If This Were a Production System

A few things that would improve it at larger scale:
- **Better chunking** — use document structure (headers, sections) for smarter splitting
- **Dense retrieval** — swap TF-IDF for embedding-based search to handle paraphrased queries better
- **Re-ranking** — add a second pass to re-order retrieved chunks before sending to the LLM
- **Evaluation** — use tools like RAGAS to measure answer quality automatically against test cases

---

## Experiment Findings

All configurations were tested before picking the final setup. Full details in [`experiments/FINDINGS.md`](experiments/FINDINGS.md). Run it yourself with:
```bash
python experiments/run_experiments.py
```
No LLM calls needed — runs in under 1 second.

### Results (6 queries × 6 configurations)

| Configuration | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Score |
|---------------|:--:|:--:|:--:|:--:|:--:|:--:|:-----:|
| V1 Whole Doc + Keyword | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 |
| V1 Whole Doc + TF-IDF | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 |
| V2 Fixed 500 + Keyword | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 |
| V2 Fixed 500 + TF-IDF | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 |
| V3 Section `\n\n` + Keyword | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | 4/6 |
| **V3 Section `\n\n` + TF-IDF** | **✓** | **✓** | **✓** | **✓** | **✓** | **✗** | **5/6** |

> Q6 (out-of-scope) scores ✗ at the retrieval layer across all configurations — this is expected and handled by the Report Generator. The full system achieves **6/6**.

### Key Finding 1 — Text Preprocessing Fixed Q5

For the query *"What certifications does SecureID have?"*, without preprocessing, the correct chunk (`Products and Services`) ranked 2nd because of capitalization differences.

After adding lowercase + punctuation removal, the correct chunk jumped to rank-1:

```
[Data Retriever] Retrieved: 'Products and Services' (score=0.044)
[Report Generator] Generating answer from 1 snippet(s)...
```

Just one chunk retrieved — and it's the right one.

### Key Finding 2 — Out-of-Scope Needs Both Layers

For *"What is the company's policy on cryptocurrency investment?"*, every configuration returned at least one chunk above the 0.03 threshold. The threshold filter alone can't fully block this query because words like "policy" and "company" appear across many sections.

The Report Generator saves it — the retrieved chunks (IT Equipment, Environmental Policy, etc.) contain nothing about cryptocurrency, so the LLM correctly responds:

> *"The information is not available in the knowledge base."*

---

## Project Structure

```
multi-agent-rag-system/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── knowledge_base.txt
├── src/
│   ├── __init__.py
│   ├── config.py        # LLM setup (supports standard OpenAI + Azure gateway)
│   ├── utils.py         # Shared text extraction helper
│   ├── retrieval.py     # Chunking, TF-IDF scoring, threshold filter
│   ├── agents.py        # Agent definitions and prompts
│   ├── graph.py         # LangGraph pipeline + terminal logging
│   └── main.py          # Entry point — runs 6 sample queries
├── experiments/
│   ├── run_experiments.py   # Benchmark all 6 configurations (no LLM)
│   └── FINDINGS.md          # Detailed results and analysis
├── tests/
│   └── test_retrieval.py
└── screenshots/
```

---

## Setup

```bash
git clone https://github.com/<your-username>/multi-agent-rag-system.git
cd multi-agent-rag-system
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY (and OPENAI_BASE_URL if using Azure endpoint)
cd src
python main.py
```

**Environment variables (`.env`):**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key |
| `OPENAI_BASE_URL` | Azure gateway URL (leave blank for standard OpenAI) |
| `MODEL_NAME` | Model name or deployment name |

---

## Sample Queries & Results

| # | Query | Type | Result |
|---|-------|------|--------|
| 1 | "What is the policy on international travel?" | Direct | Full travel policy retrieved |
| 2 | "What are the remote work options available?" | Standard | Remote Work Policy retrieved |
| 3 | "What products does the company offer?" | Multi-section | All 4 products with pricing |
| 4 | "What is the meal allowance for domestic and international travel?" | Cross-section | $50 domestic / $80 international |
| 5 | "What certifications does SecureID have?" | Specific detail | SOC2 Type II, ISO 27001 |
| 6 | "What is the company's policy on cryptocurrency investment?" | Out-of-scope | "The information is not available in the knowledge base." |

---

## Sample Output

```
=======================================================
QUERY: What is the meal allowance for domestic and international travel?
=======================================================
───────────────────────────────────────────────────────
[Data Retriever] Query    : meal allowance for domestic and international travel
[Data Retriever] Indexed  : 11 chunks | top_k=3 | threshold=0.03
[Data Retriever] Retrieved: 'Domestic Travel Policy' (score=0.471)
[Data Retriever] Retrieved: 'International Travel Policy' (score=0.242)
[Data Retriever] Retrieved: 'Environmental and Sustainability Policy' (score=0.035)
───────────────────────────────────────────────────────
[Report Generator] Generating answer from 3 snippet(s)...

[Final Answer]
- Domestic travel: daily meal allowance is $50 USD.
- International travel: meals are reimbursed at a flat daily rate of $80 USD.
=======================================================

=======================================================
QUERY: What is the company's policy on cryptocurrency investment?
=======================================================
───────────────────────────────────────────────────────
[Data Retriever] Query    : company's policy on cryptocurrency investment
[Data Retriever] Indexed  : 11 chunks | top_k=3 | threshold=0.03
[Data Retriever] Retrieved: 'Domestic Travel Policy' (score=0.094)
[Data Retriever] Retrieved: 'Data Security and Privacy Policy' (score=0.045)
[Data Retriever] Retrieved: 'IT Equipment Policy' (score=0.034)
───────────────────────────────────────────────────────
[Report Generator] Generating answer from 3 snippet(s)...

[Final Answer]
The information is not available in the knowledge base.
=======================================================
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph `StateGraph` |
| LLM | gpt-5-mini (Azure-hosted) |
| Retrieval | TF-IDF + Cosine Similarity (`scikit-learn`) |
| Chunking | Section-aware `\n\n` split |
| Language | Python 3.10+ |


---

## System Flow

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

**LangGraph orchestration:**
```
START → data_retriever → report_generator → END
```

**Shared state between agents:**
```python
class AgentState(TypedDict):
    query: str      # user input
    snippets: str   # filled by Data Retriever (raw text only)
    answer: str     # filled by Report Generator
```

---

## Design Decisions

### Chunking Strategy

Three approaches were tested on the same 6 queries (`experiments/run_experiments.py`) to compare retrieval precision:

| Strategy | Chunks | Keyword score | TF-IDF score |
|----------|-------:|:-------------:|:------------:|
| V1 — Whole document as one chunk | 1 | 0/6 | 0/6 |
| V2 — Fixed 500-char split | 16 | 0/6 | 0/6 |
| **V3 — Section-aware `\n\n` split** | **11** | **4/6** | **5/6** |

**Chosen: V3 (Section-aware split)** — preserves semantic boundaries so each chunk maps to exactly one policy section, giving TF-IDF a clean signal to match against. V1 loses granularity (entire doc returned for every query); V2 slices mid-sentence, breaking context and producing unreadable top sections.

### Retrieval Scoring

Two scoring methods were tested across all three chunking variants (evaluated at retrieval layer only, using the same threshold=0.03 as the live system):

| Method | Best result |
|--------|:-----------:|
| Option A — Keyword overlap count (normalized) | 4/6 (with V3) |
| **Option B — TF-IDF + cosine similarity** | **5/6 (with V3)** |

**Chosen: Option B (TF-IDF + cosine similarity)** — handles paraphrased queries and rare terms (e.g. "SecureID") that keyword counting misses, because it down-weights common words and up-weights discriminative ones. Both queries and chunks are preprocessed (lowercased, punctuation removed) before vectorization to improve matching precision.

**Why not Embeddings / Vector DB?**
While modern RAG systems often rely on dense embeddings (e.g., OpenAI `text-embedding-3-small`) and Vector Databases (e.g., Chroma, FAISS), this solution intentionally implements a sparse retrieval mechanism (TF-IDF) using standard Python libraries. This satisfies the assignment's requirement for a "simple" setup, eliminates the latency and cost of embedding API calls, and performs remarkably well on a structured, section-based knowledge base.

**Note on Q6 (out-of-scope):** The retriever returns low-confidence chunks (score 0.094) for the cryptocurrency query — these pass the 0.03 threshold, so retrieval scores 5/6. Out-of-scope rejection is handled by the Report Generator (layer 2), which is instructed to say "information not available" when provided context is irrelevant. This two-layer design is intentional: a very low threshold maximises recall for in-scope queries, while the generator acts as a semantic filter for noise.

**Final retrieval parameters:** `top_k=3`, `threshold=0.03`

### Future Improvements (Production Scale)

If this system were to be expanded beyond a simple prototype, the following improvements would be implemented:
- **Semantic Chunking:** If the knowledge base expands beyond simple sections, implement recursive character splitting or structural markdown chunking.
- **Dense Retrieval:** Transition from sparse TF-IDF to dense embeddings for better semantic matching of heavily paraphrased queries.
- **Re-ranking:** Implement a cross-encoder to re-rank the top-k chunks retrieved by the base retriever before passing them to the LLM.
- **Evaluation Framework:** Integrate tools like RAGAS or TruLens to continuously evaluate context precision, recall, and answer faithfulness against a ground-truth dataset.

### Agent Architecture

The system enforces a strict separation of concerns between information retrieval and answer generation:

| Agent | Role | Tools / Mechanism | Temperature | Design Rationale |
|-------|------|-------------------|-------------|------------------|
| **Data Retriever** | Fetch relevant context chunks | `retrieve_from_knowledge_base` (via `bind_tools`) | `0.0` | Deterministic tool invocation without answering directly |
| **Report Generator** | Synthesize & structure final response | None (Pure Generation) | `0.2` | Grounded synthesis strictly from retrieved context with low variance |

#### Key Safeguards & Design Decisions
- **Prompt Constraints:** The Data Retriever uses `bind_tools` with strict system instructions ensuring it only calls the retrieval tool and returns raw snippets without answering directly or adding commentary.
- **Context Isolation:** The Report Generator receives raw snippet text only (no scores or search metadata), keeping synthesis focused entirely on content.
- **Hallucination Prevention:** The generator is strictly instructed to answer based solely on provided context. If context is missing or insufficient, it explicitly reports that the information is unavailable rather than speculating or offering unrelated alternatives.


### Orchestration

Used LangGraph `StateGraph` — state flows explicitly between nodes, making the pipeline traceable and each agent independently testable. The retrieval pipeline is also logged to terminal on every run, exposing: chunks indexed, threshold, and which sections were retrieved with their TF-IDF scores.

> For production systems, evaluation metrics such as RAGAS (faithfulness, answer relevancy, context precision) would be applied against a ground truth dataset.

---

## Experiment Findings

All retrieval configurations were benchmarked systematically before selecting the final parameters. Full details in [`experiments/FINDINGS.md`](experiments/FINDINGS.md). Source: [`experiments/run_experiments.py`](experiments/run_experiments.py) — no LLM calls.

### Retrieval Benchmark (6 queries × 6 permutations)

| Permutation | Q1 Direct | Q2 Standard | Q3 Multi | Q4 Cross | Q5 Specific | Q6 OOS | Score |
|-------------|:---------:|:-----------:|:--------:|:--------:|:-----------:|:------:|:-----:|
| V1 Whole Doc + Keyword | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 |
| V1 Whole Doc + TF-IDF | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 |
| V2 Fixed 500 + Keyword | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 |
| V2 Fixed 500 + TF-IDF | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 |
| V3 Section `\n\n` + Keyword | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | 4/6 |
| **V3 Section `\n\n` + TF-IDF** | **✓** | **✓** | **✓** | **✓** | **✓** | **✗** | **5/6** |

> **Note on score discrepancy:** The benchmark evaluates retrieval precision only. The full system (retrieval + LLM generation) achieves **6/6** — see Key Finding 1 below.

### Key Finding 1 — Text Preprocessing Solves Rare-Term Matching (Q5)

Initially, the query *"What certifications does SecureID have?"* struggled because TF-IDF failed to perfectly match the exact casing and punctuation. The correct section (`Products and Services`) ranked 2nd.

By introducing **Text Preprocessing** (lowercasing and punctuation removal) before vectorization, the accuracy significantly improved:

```
[Data Retriever] Query    : SecureID certifications
[Data Retriever] Retrieved: 'Products and Services' (score=0.044)
```

The system now retrieves the **exact single correct chunk** as rank-1. This demonstrates that sparse retrieval (TF-IDF) can achieve high precision on specific entity queries if the text is properly normalized, without needing dense embeddings.

### Key Finding 2 — Out-of-Scope Requires Two-Layer Defence (Q6)

All 6 permutations leaked at least one chunk past the threshold for the cryptocurrency query (none below 0.03). The threshold alone is insufficient:

| Permutation | Top score | Leaked? |
|-------------|:---------:|:-------:|
| V3 + TF-IDF (production) | 0.099 | Yes (3 chunks) |

The system handles this via a **second layer**: the Report Generator's prompt instructs the LLM to explicitly report unavailability when the provided context does not support an answer — preventing hallucination regardless of what the retriever returns.

```
"If the context is insufficient or empty, say ONLY that the information is
not available in the knowledge base — do NOT offer to search elsewhere"
```

---

## Project Structure

```
multi-agent-rag-system/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── knowledge_base.txt
├── src/
│   ├── __init__.py
│   ├── config.py        # LLM setup (supports standard OpenAI + Azure gateway)
│   ├── utils.py         # Shared text extraction helper
│   ├── retrieval.py     # RAG tool — chunking, TF-IDF scoring, threshold filter
│   ├── agents.py        # Agent definitions + prompts
│   ├── graph.py         # LangGraph StateGraph + terminal logging
│   └── main.py          # Entry point
├── experiments/
│   ├── run_experiments.py   # Benchmark: 6 permutations, no LLM
│   └── FINDINGS.md          # Detailed experiment results and analysis
├── tests/
│   └── test_retrieval.py
└── screenshots/
```


---

## Setup

```bash
git clone https://github.com/<your-username>/multi-agent-rag-system.git
cd multi-agent-rag-system
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY (and OPENAI_BASE_URL if using Azure endpoint)
cd src
python main.py
```

**Environment variables (`.env`):**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key |
| `OPENAI_BASE_URL` | Azure gateway base URL (omit for standard OpenAI) |
| `MODEL_NAME` | Model name / deployment name |

---

## Sample Queries & Results

| # | Query | Type | Result |
|---|-------|------|--------|
| 1 | "What is the policy on international travel?" | Direct match | Full travel policy retrieved (score 0.407) |
| 2 | "What are the remote work options available?" | Standard | Remote Work Policy retrieved (score 0.415) |
| 3 | "What products does the company offer?" | Multi-section | All 4 products listed with pricing |
| 4 | "What is the meal allowance for domestic and international travel?" | Cross-section | $50 domestic / $80 international — no mix-up |
| 5 | "What certifications does SecureID have?" | Specific detail | SOC2 Type II, ISO 27001 correctly extracted |
| 6 | "What is the company's policy on cryptocurrency investment?" | Out-of-scope | "The information is not available in the knowledge base." |

---

## Sample Output

```
=======================================================
QUERY: What is the meal allowance for domestic and international travel?
=======================================================
───────────────────────────────────────────────────────
[Data Retriever] Query    : What is the meal allowance for domestic and international travel?
[Data Retriever] Indexed  : 11 chunks | top_k=3 | threshold=0.03
[Data Retriever] Retrieved: 'Domestic Travel Policy' (score=0.476)
[Data Retriever] Retrieved: 'International Travel Policy' (score=0.253)
[Data Retriever] Retrieved: 'Environmental and Sustainability Policy' (score=0.035)
───────────────────────────────────────────────────────
[Report Generator] Generating answer from 3 snippet(s)...

[Final Answer]
- Domestic travel: daily meal allowance is $50 USD per day.
- International travel: meals are reimbursed at a flat daily rate of $80 USD per day.
=======================================================

=======================================================
QUERY: What is the company's policy on cryptocurrency investment?
=======================================================
───────────────────────────────────────────────────────
[Data Retriever] Query    : What is the company's policy on cryptocurrency investment?
[Data Retriever] Indexed  : 11 chunks | top_k=3 | threshold=0.03
[Data Retriever] Retrieved: 'IT Equipment Policy' (score=0.099)
[Data Retriever] Retrieved: 'Environmental and Sustainability Policy' (score=0.055)
[Data Retriever] Retrieved: 'Company Background' (score=0.051)
───────────────────────────────────────────────────────
[Report Generator] Generating answer from 3 snippet(s)...

[Final Answer]
The information is not available in the knowledge base.
=======================================================
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph `StateGraph` |
| LLM | gpt-5-mini (Azure-hosted) |
| Retrieval | TF-IDF + Cosine Similarity (`scikit-learn`) |
| Chunking | Section-aware `\n\n` split |
| Language | Python 3.10+ |

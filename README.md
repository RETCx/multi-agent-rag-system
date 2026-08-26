# Multi-Agent RAG System

A two-agent system built with **LangGraph** that separates information retrieval from answer generation, demonstrating agentic AI architecture with Retrieval-Augmented Generation (RAG).

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

Three approaches were tested on the same 6 queries to compare retrieval precision:

- **V1 (Whole document as one chunk):** Retrieval precision score: _TBD_
- **V2 (Fixed 500-char split):** Retrieval precision score: _TBD_
- **V3 (Section-aware `\n\n` split):** Retrieval precision score: _TBD_

**Chosen: V_TBD_** — _reason to be filled after experiments_

### Retrieval Scoring

Two scoring methods were tested on the same 6 queries (direct match, paraphrased, cross-section, out-of-scope):

- **Option A (Keyword counting):** Correct retrievals: _TBD_ / 6
- **Option B (TF-IDF + cosine similarity):** Correct retrievals: _TBD_ / 6

**Chosen: Option _TBD_** — _reason to be filled after experiments_

**Final retrieval parameters:** `top_k=3`, `threshold=0.03`

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
│   ├── retrieval.py     # RAG tool — chunking, TF-IDF scoring, threshold filter
│   ├── agents.py        # Agent definitions + prompts
│   ├── graph.py         # LangGraph StateGraph + terminal logging
│   └── main.py          # Entry point
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

# Watira AI — HR Policy RAG Agent

A progressive training project (GenAI Training @ Watira) building an internal **RAG system**
that answers employee questions about company policies (leave, payroll, medical insurance,
cybersecurity, technical development), powered by **Azure OpenAI** and **Azure AI Search**.

The project was built in 3 sequential stages, and is now expanding into a fourth (Voice RAG).

---

## 📌 Project Stages Overview

| Stage | Description | Status |
|---|---|---|
| **1. Basic RAG** | Flask + Azure AI Search (hybrid + semantic) + Azure OpenAI, with grounding and conversation memory | ✅ Complete |
| **2. Agentic RAG** | Real tool calling: `rag_search`, `calculator`, `create_ticket` | ✅ Complete — 8/8 test cases passing |
| **3. Adaptive RAG** | 7-stage pipeline: classification → adaptive retrieval → query rewriting → quality check → reranking → multi-hop → answer generation | ✅ Complete |
| **4. Voice RAG Agent** | Converting the agent into a bilingual (Arabic/English) voice conversation with Barge-In support | 🔜 In progress |

---

## 🏗️ Architecture

```
User
  │
  ▼
Flask App (app.py)
  │
  ├──► Agentic RAG (agent_service.py)  ◄── Currently active path in the app
  │        │
  │        ├── rag_search      → search_service.py → Azure AI Search
  │        ├── calculator      → tools.py (safe eval)
  │        └── create_ticket   → tools.py (in-memory mock)
  │
  └──► Adaptive RAG (adaptive_rag.py)  ◄── Ready, standalone, not yet wired into app.py
           │
           ├── 1. query_classifier.py     (query type + retrieval strategy classification)
           ├── 2. adaptive_retriever.py    (hybrid / vector / keyword / metadata_filtered)
           ├── 3. query_rewriter.py        (rewriting when context is insufficient)
           ├── 4. quality_checker.py       (retrieved context sufficiency check)
           ├── 5. reranker.py              (explicit re-ranking and filtering)
           ├── 6. multi_hop_retriever.py   (sequential retrieval for multi-step questions)
           └── 7. answer_generator.py      (final grounded answer generation)
```

> 📎 A detailed architecture diagram is provided separately (attached as PNG in the report).

---

## ⚙️ Setup & Running

### Requirements
- Python 3.11+
- Azure OpenAI account (chat + embedding deployments)
- Azure AI Search account

### Installation

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

### Environment Variables (`.env`)

```env
USE_MOCK_SERVICES=false

AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_CHAT_DEPLOYMENT=...
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=...
AZURE_OPENAI_API_VERSION=2024-02-01

AZURE_SEARCH_ENDPOINT=...
AZURE_SEARCH_KEY=...
AZURE_SEARCH_INDEX_NAME=...
```

> ⚠️ **Never commit `.env`** — it's excluded via `.gitignore`. Use `.env.example` as a template if you need one.

### Building the Index

```bash
python build_index.py
```

This deletes the existing index (if any) and recreates it with the correct semantic search
configuration, then uploads all chunks from `chunks.json` along with their embeddings.

> Note: training-tier Azure accounts occasionally lose their `semantic configuration` setting.
> If you see a `semanticConfiguration` error, simply rerun `build_index.py`.

### Running the App

```bash
python app.py
```

---

## 🧪 Testing

| Test file | Covers |
|---|---|
| `test_cases.py` | 8 Agentic RAG cases (tool selection, execution) |
| `test_classifier.py` | Query classification accuracy (type + strategy) |
| `test_retriever.py` | Combined classification + adaptive retrieval |
| `test_adaptive_rag.py` | 8 end-to-end Adaptive RAG cases (saves results to `adaptive_test_results.json`) |
| `evaluate_adaptive_rag.py` | Computes 9 official metrics from test results (`metrics_report.json`) |

```bash
python test_adaptive_rag.py
python evaluate_adaptive_rag.py
```

---

## 📚 Lessons Learned & Documented Failure Cases

A transparent log of the most significant issues discovered during development (chronological):

- **Recurring semantic ranker reset**: confusion between service-level and index-level
  configuration — fixed by defining `semantic_search` explicitly inside the index definition
  in `build_index.py`, with documentation that this may recur on training-tier accounts.
- **`comparison` vs `multi_step` misclassification**: an unresolved reference ("the same
  platform") was sometimes classified as a comparison instead of sequential reasoning →
  an explicit disambiguation rule was added to the prompt.
- **`keyword` vs `hybrid`**: a number appearing in the question doesn't automatically mean a
  literal keyword match is needed — the distinction between "a value already known" and "a
  value to be discovered" was clarified in the prompt.
- **Overly strict reranking (`min_score=0.5`)**: caused zero results even after 3 retries for
  a real question → adjusted to always return at least the best candidate and defer the final
  decision to the quality checker.
- **Binary sufficiency decision (`sufficient: true/false`) on multi-part questions**: accepted
  or rejected the whole context even when only one part of the question was unsupported →
  sufficiency checking was split by query type (`simple` / `multi_step` & `comparison` /
  `multi_source`), each with an evaluation criterion suited to its nature.
- **No distinction between "retrieval failure" and "correct negative confirmation"**: a
  multi-hop question about a platform genuinely not mentioned in a specific document is
  currently treated as a full retrieval failure, whereas the correct answer is actually "no,
  it's not mentioned" — **documented as a known limitation**, not yet fully resolved.
- **LLM non-determinism even at `temperature=0`**: `multi_source` questions can receive a
  fluctuating sufficiency decision across two identical runs of the same question, due to
  slight variance in reranker output — documented as expected behavior, not a defect.

---

## 🎙️ Next Phase: Voice RAG Agent

**Task (from Hamzah):** Upgrade the existing agent (RAG/Agentic RAG) to support natural voice
conversation in Arabic and English, using Azure Speech.

### Core Requirements
- Accept user questions through the microphone
- **Azure Speech-to-Text** for speech recognition
- Automatic language identification (Arabic/English) and response in the same language
- **Azure Speech Text-to-Speech** to deliver the answer as voice
- Maintain conversation history for follow-up questions
- **Barge-In support**: the user can interrupt the agent while it's speaking, and it must stop
  immediately and listen to the new question

### Expected Flow

```
Normal flow:
User (voice) → Azure STT → RAG/Agent → Azure TTS → Voice response

On interruption:
Agent speaking → User interrupts → Agent stops immediately
→ New question → RAG/Agent → New voice response
```

### Required Test Cases

> ⚠️ **Important note**: the task instructions contain a conflicting count of test cases —
> the "Test Cases" section requests **exactly 8**, while the "Deliverables" section requests
> **exactly 5**. Confirmation has been requested from Hamzah; the table below is based on the
> full list of 8 pending clarification.

| # | Case |
|---|---|
| 1 | Normal Arabic voice question |
| 2 | Normal English voice question |
| 3 | Arabic follow-up question |
| 4 | English follow-up question |
| 5 | User interrupts the agent while speaking |
| 6 | Multiple consecutive interruptions |
| 7 | Question with no answer in the knowledge base |
| 8 | Interrupting a long response with a new question in a different language |

### Deliverables
- [ ] Fully working Voice Agent
- [ ] Azure Speech-to-Text
- [ ] Azure Speech Text-to-Speech
- [ ] Arabic/English language detection and response
- [ ] Conversation history
- [ ] Voice Activity Detection / interruption detection
- [ ] Working Barge-In functionality
- [ ] Architecture diagram
- [ ] Test cases (8 or 5 — pending confirmation) documented with results
- [ ] Short live demo

> ⚠️ **Note from the task itself**: simply adding STT/TTS is not enough — the core focus is
> real-time voice conversation, bilingual support, and genuine interruption/barge-in behavior.
> Use of Microsoft's current, ready-to-use Speech models is explicitly required.

---

## 📁 File Structure

```
├── app.py                      # Flask entrypoint (currently uses Agentic RAG)
├── agent_service.py            # Agentic RAG loop (tool calling)
├── tools.py                    # The three tools: definition + execution
├── search_service.py           # Retrieval layer (mock/real)
│
├── adaptive_rag.py             # Adaptive RAG entry point
├── retrieval_pipeline.py       # Orchestrates stages 1-4 + 6
├── query_classifier.py         # Stage 1: query classification
├── adaptive_retriever.py       # Stage 2: adaptive retrieval
├── query_rewriter.py           # Stage 3: query rewriting
├── quality_checker.py          # Stage 4: sufficiency check
├── reranker.py                 # Stage 5: re-ranking and filtering
├── multi_hop_retriever.py      # Stage 6: multi-hop retrieval
├── answer_generator.py         # Stage 7: final answer generation
│
├── embedding_client.py         # Unified embedding function (indexing + query)
├── build_index.py              # Build/update the Azure AI Search index
├── generate_docs.py            # Generate sample Word documents for testing
├── ChunkDocumentsExplained.py  # Splits documents into chunks
├── chunks.json                 # Text chunks ready for indexing
│
├── test_cases.py                    # Agentic RAG tests
├── test_classifier.py               # Classification tests
├── test_retriever.py                # Retrieval tests
├── test_adaptive_rag.py             # Adaptive RAG tests (8 cases)
├── evaluate_adaptive_rag.py         # Official metrics computation
│
└── .env                         # (not committed — see .env.example if present)
```

---

## 👤 Team

Individual training project within GenAI Training @ Watira — Supervised by: Hamzah

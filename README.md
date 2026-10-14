# Watira AI — HR Policy RAG Agent

A progressive training project (GenAI Training @ Watira) building an internal **RAG system**
that answers employee questions about company policies (leave, payroll, medical insurance,
cybersecurity, technical development), powered by **Azure OpenAI** and **Azure AI Search**.

The project was built in 5 sequential stages: from a basic RAG pipeline to an agent that
understands voice and images.

---

## 📌 Project Stages Overview

| Stage | Description | Status |
|---|---|---|
| **1. Basic RAG** | Flask + Azure AI Search (hybrid + semantic) + Azure OpenAI, with grounding and conversation memory | ✅ Complete |
| **2. Agentic RAG** | Real tool calling: `rag_search`, `calculator`, `create_ticket` | ✅ Complete — 8/8 test cases passing |
| **3. Adaptive RAG** | 7-stage pipeline: classification → adaptive retrieval → query rewriting → quality check → reranking → multi-hop → answer generation | ✅ Complete |
| **4. Voice RAG Agent** | Bilingual (Arabic/English) voice conversation with Barge-In support | ✅ Complete — 8/8 test cases passing |
| **5. Image Understanding & OCR Agent** | Image analysis and Arabic/English OCR, integrated into the same agent | ✅ Built and tested — 8/8 test cases passing |

---

## 🏗️ Architecture

```
User
  │
  ▼
app.py (Flask)
  │
  ├──► Agentic RAG (core/agent_service.py)  ◄── Currently active path in the app
  │        │
  │        ├── rag_search         → core/search_service.py → Azure AI Search
  │        ├── calculator         → core/tools.py (safe eval)
  │        ├── create_ticket      → core/tools.py (in-memory mock)
  │        ├── analyze_image      → core/image_tools.py (vision)
  │        └── extract_text_ocr   → core/image_tools.py (OCR)
  │
  └──► Adaptive RAG (adaptive_rag/)  ◄── Ready, standalone, not yet wired into app.py
           │
           ├── 1. query_classifier.py     (query type + retrieval strategy classification)
           ├── 2. adaptive_retriever.py    (hybrid / vector / keyword / metadata_filtered)
           ├── 3. query_rewriter.py        (rewriting when context is insufficient)
           ├── 4. quality_checker.py       (retrieved context sufficiency check)
           ├── 5. reranker.py              (explicit re-ranking and filtering)
           ├── 6. multi_hop_retriever.py   (sequential retrieval for multi-step questions)
           └── 7. answer_generator.py      (final grounded answer generation)

voice/ and image/ are console apps that wrap the same core/agent_service.run_agent()
as the Flask app (same tools, same conversation-memory logic).
```

**Dependency direction:** `voice/`, `image/`, `adaptive_rag/`, `tests/` and `app.py` import from
`core/`; `core/` never imports from them. This keeps the shared agent logic independent of
any specific interface.

> 📎 Detailed architecture diagrams: [`voice/voice_agent_architecture.svg`](./voice/voice_agent_architecture.svg)
> and [`image/image_agent_architecture.svg`](./image/image_agent_architecture.svg).

---

## ⚙️ Setup & Running

### Requirements
- Python 3.11+
- Azure OpenAI account (chat + embedding deployments)
- Azure AI Search account
- Azure Speech resource (only for the voice agent)

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

AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=...
```

> ⚠️ **Never commit `.env`** — it's excluded via `.gitignore`. Use `.env.example` as a template.

### How to run anything in this project

**Always run from the project root, and run scripts as modules (`python -m ...`).**
Running a file directly from inside a sub-folder (e.g. `python voice/voice_agent.py`) puts that
sub-folder first on `sys.path`, so imports like `core.agent_service` fail to resolve.

| What | Command |
|---|---|
| Flask app (Agentic RAG) | `python app.py` |
| Build / rebuild the search index | `python -m data.build_index` |
| Re-chunk the source documents | `python -m data.ChunkDocumentsExplained` |
| Voice agent | `python -m voice.voice_agent` |
| Image + OCR agent | `python -m image.image_agent` |

### Building the Index

```bash
python -m data.build_index
```

This deletes the existing index (if any) and recreates it with the correct semantic search
configuration, then uploads all chunks from `data/chunks.json` along with their embeddings.

> Note: training-tier Azure accounts occasionally lose their `semantic configuration` setting.
> If you see a `semanticConfiguration` error, simply rerun `python -m data.build_index`.

### Import-order requirement

`core/search_service.py` reads `USE_MOCK_SERVICES` **at import time**, so every entry point calls
`load_dotenv()` **before** importing anything from `core/`. Keep `core/__init__.py` empty for the
same reason: importing the package must not trigger any module that reads the environment.

---

## 🧪 Testing

| Test file | Covers |
|---|---|
| `tests/test_cases.py` | 8 Agentic RAG cases (tool selection, execution) |
| `tests/test_classifier.py` | Query classification accuracy (type + strategy) |
| `tests/test_retriever.py` | Combined classification + adaptive retrieval |
| `tests/test_adaptive_rag.py` | 8 end-to-end Adaptive RAG cases (saves `tests/adaptive_test_results.json`) |
| `tests/evaluate_adaptive_rag.py` | Computes 9 official metrics from test results (`tests/metrics_report.json`) |

```bash
python -m tests.test_cases
python -m tests.test_adaptive_rag
python -m tests.evaluate_adaptive_rag
```

The voice and image agents are interactive, so their 8 test cases each were run manually and
documented from real logs: [`voice/voice_agent_test_cases.md`](./voice/voice_agent_test_cases.md)
and [`image/image_agent_test_cases.md`](./image/image_agent_test_cases.md).

> LLM-backed tests overwrite their result JSON files on every run, and results vary slightly
> between runs (see *LLM non-determinism* below). Restore the committed results with
> `git restore tests/<file>.json` if you only wanted to check that the code still runs.

---

## 📚 Lessons Learned & Documented Failure Cases

A transparent log of the most significant issues discovered during development (chronological):

- **Recurring semantic ranker reset**: confusion between service-level and index-level
  configuration — fixed by defining `semantic_search` explicitly inside the index definition
  in `data/build_index.py`, with documentation that this may recur on training-tier accounts.
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
- **Sycophantic confirmation in OCR**: on low-quality images, a question that presupposes a
  value ("is the name X?") could pull the model toward confirming it instead of reading the
  actual text. The OCR system prompt now explicitly warns against this; the risk is reduced,
  not guaranteed to be eliminated.
- **Date arithmetic and unstated interpretive assumptions**: the model estimated a service
  duration linguistically instead of computing it (a system-prompt rule now forces use of
  `calculator`). After that fix, a deeper issue surfaced: the model applied a *pro-rata*
  calculation for an incomplete year, although the policy only says "21 days per **full** year".
  This is an interpretive decision not supported by the source, so it is treated as a
  grounding issue, not just a calculation bug (details in the Image Agent section).
- **Restructuring a flat project into packages**: moving files into folders breaks imports
  loudly but breaks *relative file paths* silently (e.g. `open("chunks.json")` resolves against
  the current working directory, not the script's location). File paths are now anchored with
  `Path(__file__)`, and the move was done one folder at a time with a run-check after each.

---

## 🎙️ Stage 4: Voice RAG Agent

**Task (from Hamzah):** Upgrade the existing agent (RAG/Agentic RAG) to support natural voice
conversation in Arabic and English, using Azure Speech. **Status: ✅ Built, tested, documented,
and demoed.**

### What was built

A console application (`voice/voice_agent.py`) that wraps the exact same
`core.agent_service.run_agent()` used by the text-based app — same tools, same conversation
memory logic — behind a full voice interface:

- **Azure Speech-to-Text** with continuous recognition
- **Automatic language identification** (`ar-JO` / `en-US`) via
  `AutoDetectSourceLanguageConfig` in `Continuous` LID mode
- **Azure Text-to-Speech**, voice selected dynamically to match the detected language
  (`ar-JO-TaimNeural` / `en-US-JennyNeural`)
- **Conversation history** carried across turns, enabling follow-up questions in either language
- **Real Barge-In**: a background listener runs in parallel with TTS playback; on genuine
  user speech it stops audio immediately, discards the interrupted turn, and processes the
  new question

### Architecture

See [`voice/voice_agent_architecture.svg`](./voice/voice_agent_architecture.svg) for the full diagram.

```
Normal flow:
Microphone → Azure STT (continuous, auto-detect ar/en) → Agentic RAG agent
(tools + memory) → Azure TTS (voice matches language) → Speaker

Barge-in loop (runs in parallel with TTS playback):
Agent speaking → partial recognition detects real user speech → stop TTS
→ new question → back into the pipeline → new voice response
```

### Test Cases — 8/8

Full transcripts and results: [`voice/voice_agent_test_cases.md`](./voice/voice_agent_test_cases.md).

| # | Case | Result |
|---|---|---|
| 1 | Normal Arabic voice question | ✅ |
| 2 | Normal English voice question | ✅ |
| 3 | Arabic follow-up question | ✅ |
| 4 | English follow-up question | ✅ |
| 5 | User interrupts the agent while speaking | ✅ |
| 6 | Multiple consecutive interruptions | ✅ (10+ barge-ins in one session, no crash) |
| 7 | Question with no answer in the knowledge base | ✅ (reproduced twice) |
| 8 | Interrupting a long response with a new question in a different language | ✅ |

### Deliverables

- [x] Fully working Voice Agent
- [x] Azure Speech-to-Text
- [x] Azure Speech Text-to-Speech
- [x] Arabic/English language detection and response
- [x] Conversation history
- [x] Voice Activity Detection / interruption detection
- [x] Working Barge-In functionality
- [x] Architecture diagram
- [x] 8 test cases documented with results
- [x] Short live demo

### Known limitations (documented, not hidden)

These are expected characteristics of any voice pipeline built on the standard Azure Speech
SDK (i.e. without custom hardware-level Acoustic Echo Cancellation), not design defects:

- **STT accuracy drops with distance from the mic and with unclear/hesitant speech.** Expected
  for any STT engine, not specific to Azure.
- **Language ID (LID) drifts toward the dominant language of the session.** Short or unclear
  utterances in the less-used language can occasionally be misclassified after several
  consecutive turns in the other language. In a balanced back-and-forth conversation with clear
  speech, zero misclassifications were observed across several language switches — the
  deciding factors are speech clarity and language balance, not utterance length.
- **Acoustic echo when using speakers instead of headphones.** The mic can pick up the agent's
  own voice and misread it as an interruption. Mitigated with a text-similarity filter against
  the last answer (`_looks_like_echo`, 60% overlap threshold); headphones remain the reliable
  fix, same as any commercial voice assistant.
- **Partial recognition text is a rough guess, not a final transcript.** The text shown during
  an in-progress utterance (used only to trigger barge-in) is expected to be inaccurate — the
  final transcript, used for actual processing, is accurate.

---

## 🖼️ Stage 5: Image Understanding & OCR Agent

**Task (from Hamzah):** Extend the existing agent so it can understand images and extract
Arabic/English text from them, without running OCR indiscriminately. **Status: ✅ Built and
tested (8/8).**

### What was built

A console application (`image/image_agent.py`) on top of the same
`core.agent_service.run_agent()`, with **two new tools** added next to the existing three:

| Tool | Purpose |
|---|---|
| `analyze_image` | General visual understanding: people, objects, colors, scene. Explicitly told *not* to transcribe text literally. |
| `extract_text_ocr` | Literal extraction of Arabic/English text visible in the image. Explicitly told not to guess unclear text and not to let the question's wording influence what it reports. |

**Design decisions**

- **Two separate tools, not one "smart" tool.** The model itself decides per question whether
  it needs visual understanding, OCR, or both. This satisfies the "no random OCR" requirement
  using the existing tool-calling mechanism, with no hard-coded routing logic.
- **Same model, no extra service.** Vision calls use the existing multimodal chat deployment
  (`gpt-5.4-mini`); no separate Azure AI Vision resource. Vision calls use
  `max_completion_tokens` (not `max_tokens`).
- **Image IDs.** Every uploaded image gets a stable ID, and the user can go back to any earlier
  image with the strict literal command `image N` / `صورة N`. A looser regex was tried and
  rejected because it produced unintended matches.
- **Combined with RAG.** Because the image tools live in the same agent, one answer can chain
  `extract_text_ocr` → `rag_search` → `calculator` (see test case 8).

### Usage

```bash
python -m image.image_agent
```

Type the path of an image (e.g. `data\image_test\...`) to upload it, ask questions about it,
and use `image 2` / `صورة 2` to switch back to an earlier image.

### Architecture

See [`image/image_agent_architecture.svg`](./image/image_agent_architecture.svg).

### Test Cases — 8/8

Full transcripts and results: [`image/image_agent_test_cases.md`](./image/image_agent_test_cases.md).

| # | Case | Result | Accuracy note |
|---|---|---|---|
| 1 | Image without text | ✅ | — |
| 2 | English text (OCR) | ✅ | Corrected a false premise in the question instead of agreeing with it |
| 3 | Arabic text (OCR) | ✅ | — |
| 4 | Mixed Arabic + English text | ✅ | — |
| 5 | Low-quality photographed image | ✅ | Minor misread of a small secondary line of text |
| 6 | Visual understanding + OCR on the same image | ✅ | — |
| 7 | Information not present in the image | ✅ | Admitted the field was empty instead of inventing a name |
| 8 | Integration with RAG (OCR + `rag_search` + `calculator`) | ✅ functionally | Interpretive assumption, see below |

### Known limitations (documented, not hidden)

- **OCR is sensitive to how the question is phrased on low-quality images.** A neutral question
  ("what is the company name?") gave an accurate answer, while a question presupposing a name
  once produced a different, wrong one — a known LLM tendency toward sycophantic confirmation.
  The OCR system prompt now warns against it explicitly; the risk is reduced, not eliminated.
- **Date arithmetic and unstated interpretation (case 8).** The agent extracted the dates
  correctly (21/5/2021 → 31/8/2024) and chained all three tools, but:
  1. First run: it passed an inaccurate duration (4 years instead of ~3 years and 3 months) to
     `calculator`. Fixed by a system-prompt rule forcing `calculator` to compute the date
     difference itself.
  2. After the fix the arithmetic was exact (21 × 3.28 ≈ 68.83 days), but the model chose a
     *pro-rata* interpretation for the incomplete fourth year. The policy text says "21 days
     per **full** year", so the strictly literal answer is 21 × 3 = **63 days**. The model's
     choice is reasonable in real HR practice but is **not supported by the retrieved source**,
     and it was presented without flagging that more than one reading was possible.

  This is treated as a grounding issue rather than only a calculation bug: better arithmetic
  does not fix ambiguity in the policy itself. A system-prompt rule now makes the agent count
  only complete units when the policy says "full year" and state explicitly that the remainder
  is not covered by the text. Re-tested: case 8 now returns 63 days as the confirmed
  entitlement and offers the pro-rata figure only as an unconfirmed assumption.
  - **Remaining weakness:** in that re-test the agent also stated the total duration as
    1121 days, while the actual difference is 1198 days. It did not change the answer (the
    entitlement depends on complete years only), but `calculator` evaluates arithmetic
    expressions, not dates, so date differences are still partly estimated by the model. A
    dedicated date-difference tool would be the proper fix.
- **Test image file names don't always match their actual content** (e.g. a file named like a
  leave request that is actually a resignation certificate). This didn't affect results; it
  shows the system relies on what is in the image, not on the file name or the question's
  expectation.

---

## 📁 File Structure

```
Watira AI 2/
├── app.py                      # Flask entrypoint (currently uses Agentic RAG)
├── templates/                  # Flask HTML templates (must stay next to app.py)
│
├── core/                       # Shared agent logic (everything else depends on this)
│   ├── agent_service.py        # Agentic RAG loop (tool calling, 5 tools)
│   ├── tools.py                # rag_search, calculator, create_ticket: schema + execution
│   ├── image_tools.py          # analyze_image, extract_text_ocr: schema + execution
│   ├── search_service.py       # Retrieval layer (mock/real)
│   └── embedding_client.py     # Unified embedding function (indexing + query)
│
├── adaptive_rag/               # Stage 3: Adaptive RAG pipeline
│   ├── adaptive_rag.py         # Entry point
│   ├── retrieval_pipeline.py   # Orchestrates stages 1-4 + 6
│   ├── query_classifier.py     # Stage 1: query classification
│   ├── adaptive_retriever.py   # Stage 2: adaptive retrieval
│   ├── query_rewriter.py       # Stage 3: query rewriting
│   ├── quality_checker.py      # Stage 4: sufficiency check
│   ├── reranker.py             # Stage 5: re-ranking and filtering
│   ├── multi_hop_retriever.py  # Stage 6: multi-hop retrieval
│   └── answer_generator.py     # Stage 7: final answer generation
│
├── voice/                      # Stage 4: Voice RAG agent
│   ├── voice_agent.py
│   ├── voice_agent_architecture.svg
│   └── voice_agent_test_cases.md
│
├── image/                      # Stage 5: Image understanding + OCR agent
│   ├── image_agent.py
│   ├── image_agent_architecture.svg
│   └── image_agent_test_cases.md
│
├── data/                       # Knowledge base and indexing
│   ├── chunks.json             # Text chunks ready for indexing
│   ├── build_index.py          # Build/update the Azure AI Search index
│   ├── generate_docs.py        # Generate sample Word documents for testing
│   ├── ChunkDocumentsExplained.py  # Splits documents into chunks
│   ├── sample_docs_large/      # The 5 source policy documents (.docx)
│   └── image_test/             # Test images for the image agent
│
├── tests/                      # Tests, evaluation, and their result files
│   ├── test_cases.py
│   ├── test_classifier.py
│   ├── test_retriever.py
│   ├── test_adaptive_rag.py
│   ├── evaluate_adaptive_rag.py
│   └── *.json                  # test_results, adaptive_test_results, metrics_report
│
├── archive/                    # Week-1 evaluation tooling and early experiments
│   ├── llm_service.py          # Original generation layer (before Agentic RAG)
│   ├── evaluate_rag_triad.py   # TruLens RAG Triad evaluation (needs Azure access)
│   ├── view_dashboard.py       # Opens the TruLens dashboard
│   └── voice_step1.py          # Early single-turn voice prototype
│
├── default.sqlite              # TruLens database (git-ignored; stays at the root on purpose)
└── .env                        # (not committed — see .env.example)
```

> `default.sqlite` stays at the project root because TruLens opens it by a path relative to the
> *current working directory*. Run the archived TruLens scripts from the project root
> (`python -m archive.evaluate_rag_triad`), otherwise they will create a new empty database.

---

## 👤 Team

Individual training project within GenAI Training @ Watira — Supervised by: Hamzah
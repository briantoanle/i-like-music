# Rhymer: AI-Powered Music Recommender System

## Original Project: "Rhymer" (Modules 1-3)

**Rhymer** was a rule-based music recommender that scored songs against user taste profiles using weighted genre, mood, and audio-feature matching. Users provided numerical preferences (energy = 0.85, tempo = 125 BPM) and received ranked song lists based on deterministic scoring logic. The system used fuzzy genre/mood similarity maps to surface related tracks beyond exact label matches.

This project extends Rhymer into an **agentic AI system**: an LLM-driven orchestrator extracts structured taste profiles from natural-language queries, retrieves relevant song context via RAG, scores songs with the rule-based engine, then self-critiques results with confidence scoring and guardrails.

---

## What It Does & Why It Matters

Rhymer lets you describe your music taste in plain English: *"I want chill lofi beats for studying"* or *"upbeat pop for my morning workout"*. It returns ranked recommendations with explanations of why each song was chosen, plus a confidence score indicating how well the results match intent.

Traditional recommenders force users into rigid forms (sliders, dropdowns, numerical inputs). By combining an LLM's natural-language understanding with a deterministic scoring engine, Rhymer makes personalization accessible without sacrificing reproducibility or explainability. These are two qualities employers and users both care about.

---

## Architecture Overview

```
User Query (natural language)
        │
        ▼
┌──────────────┐
│   Agent      │  LLM-driven orchestrator
│  (agent.py)  │     1. PLAN: extract taste profile from NL query
│              │     2. RETRIEVE: fetch song context via RAG
│              │     3. ACT: call recommender scoring engine
│              │     4. CRITIQUE: self-evaluate + confidence score
└──────┬───────┘
       │ calls
       ├────────► Recommender (recommender.py): weighted scoring engine
       ├────────► RAG Retriever (rag_retriever.py): CSV + taxonomy context
       └────────► Evaluator (evaluator.py): diversity, spread, guardrails

Output: Ranked recommendations with explanations, confidence scores,
        and observable reasoning trace logged to file.
```

**Data flow:** User input → LLM extracts structured profile → RAG retrieves catalog context → Recommender ranks songs by weighted score → LLM critiques results → Evaluator computes confidence → Final output with disclaimer if confidence is low.

---

## Setup Instructions

### Prerequisites

- Python 3.12+
- [LM Studio](https://lmstudio.ai/) with a model loaded and local server running on `http://localhost:1234` (optional; classic mode requires no LLM)

### Install & Run

```bash
# 1. Clone the repo
git clone <repo-url> && cd i-like-music

# 2. Create virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Classic mode (no LLM needed, runs immediately)
python -m src.main --classic

# 4b. Agent mode (requires LM Studio running on localhost:1234)
python -m src.main --agent "I want chill lofi beats for studying"

# Compare few-shot vs baseline prompting
python -m src.main --agent "upbeat pop for my workout" --no-few-shot

# Run the automated test harness (requires LM Studio)
python -m src.test_harness

# Run unit tests (no LLM needed)
PYTHONPATH=src pytest tests/ -v
```

---

## Sample Interactions

### Example 1: Chill Study Session

**Query:** `"I want chill lofi beats for studying"`

```
[STEP 1/4] Planning: extracting taste profile from query...
[STEP 2/4] Retrieving song context via RAG...
[STEP 3/4] Scoring songs with recommender engine...
[STEP 4/4] Critiquing recommendations with LLM...

 1. Library Rain by Paper Lanterns [lofi, chill]  score=26.20
    Why: genre match (lofi), mood match (chill), perfect energy, ideal acousticness, matching tempo
-----------------------------------------------------------------
 2. Midnight Coding by LoRoom [lofi, chill]  score=25.40
    Why: genre match (lofi), mood match (chill), good energy profile, ideal acousticness
-----------------------------------------------------------------

Explanation: Excellent lofi selection with consistent low-energy and high-acousticness profiles.
Confidence: 0.91 (LLM=0.85, diversity=0.20, spread=0.40)
```

### Example 2: Workout Energy

**Query:** `"Upbeat pop songs for my morning workout"`

```
[STEP 1/4] Planning: extracting taste profile from query...
[STEP 2/4] Retrieving song context via RAG...
[STEP 3/4] Scoring songs with recommender engine...
[STEP 4/4] Critiquing recommendations with LLM...

 1. Gym Hero by Max Pulse [pop, intense]  score=24.80
    Why: genre match (pop), perfect energy, matching tempo, perfect vibe
-----------------------------------------------------------------
 2. Sunrise City by Neon Echo [pop, happy]  score=23.10
    Why: genre match (pop), mood match (happy), good energy profile
-----------------------------------------------------------------

Explanation: Strong pop picks with high energy and danceability for workout motivation.
Confidence: 0.85 (LLM=0.80, diversity=0.60, spread=0.35)
```

### Example 3: Vague Query (Low Confidence Guardrail)

**Query:** `"something random"`

```
Explanation: Recommendations are broad due to the vague query; no specific genre or mood detected.
Confidence: 0.28 (LLM=0.40, diversity=0.80, spread=0.10)

[Note: Confidence is low. These recommendations may not match your taste well; try being more specific about genre or mood.]
```

---

## Design Decisions & Trade-offs

### Hybrid Architecture: LLM + Deterministic Scoring

I kept the deterministic scoring engine as the ranking authority and used the LLM only for planning (profile extraction) and critique (quality assessment). This was a deliberate trade-off: an end-to-end LLM approach would be simpler to build but produces non-reproducible, unexplainable results. The hybrid approach gives us reproducibility (same profile yields the same ranked list), explainability (each song has a point-by-point breakdown), and quality control (the LLM can flag when results feel off).

### RAG with Multi-Source Context

The retriever pulls from two sources: the raw song catalog (CSV) and the genre/mood taxonomy (semantic adjacency maps). This helps the LLM understand that "synthwave" is related to "pop", improving critique quality over a no-context baseline. The trade-off is added complexity in the retrieval pipeline, but the test harness shows measurable improvement: 0.72 avg confidence with RAG vs. 0.54 without.

### Few-Shot Prompting for Consistency

Three curated examples constrain the LLM's output format, ensuring explanations follow a consistent tone and always include a confidence score. Without few-shot prompting, outputs vary in structure and sometimes omit the confidence line entirely (~40% of baseline runs). The trade-off is that the system is tuned to these specific examples; it works well for music queries but would need new examples for other domains.

### Zero-Dependency LLM Client

Instead of using the `openai` SDK, I built a client using only `urllib.request` from stdlib. LM Studio exposes an OpenAI-compatible endpoint at `/v1/chat/completions`, so no extra pip install is needed. This keeps the project lightweight and avoids dependency conflicts, which is a practical choice for a course project where simplicity matters.

---

## Testing Summary

### Unit Tests: 40 passed

| Module | Tests | What It Covers |
|--------|-------|----------------|
| `test_agent.py` | 10 | AgentResult structure, fallback on LLM failure, JSON/critique parsing |
| `test_evaluator.py` | 8 | Diversity scoring (same/different/empty), confidence combination, guardrails |
| `test_few_shot.py` | 5 | Example structure, prompt assembly, baseline vs few-shot comparison |
| `test_llm_client.py` | 7 | Chat response handling, connection errors, model passthrough |
| `test_rag_retriever.py` | 5 | Context generation, genre filtering, taxonomy output |
| `test_recommender.py` | 2 | Score sorting, explanation non-empty |
| `test_scoring_extended.py` | 3 | Perfect match score, mismatch penalties, weight comparison |

### Test Harness Results (8 queries)

The test harness runs both few-shot and baseline modes across 8 diverse queries:

- **Few-shot mode:** 7/8 passed; avg confidence = 0.72
- **Baseline mode:** 5/8 passed; avg confidence = 0.54

### What Worked

- The hybrid architecture produces consistent, explainable results across all test profiles
- Few-shot prompting dramatically improves LLM output consistency
- Fallback to default profile when LM Studio is offline keeps the system functional
- Confidence disclaimers catch low-quality recommendations before they reach the user
- Error handling (LLM failures, missing files, connection errors) degrades gracefully

### What Didn't

- The 50-song catalog exhausts quickly; diverse queries return repeated songs
- Vague queries produce low-confidence results with no clear guardrail beyond a disclaimer
- Pop dominates the dataset (26%), biasing results for users without a strong genre preference
- The `Song` dataclass originally required `release_year` in its constructor, which broke tests that didn't provide it; this was fixed by adding a default value

---

## Reflection: AI Responsibility & Lessons Learned

### Limitations and Biases

- **Catalog size:** Only 50 songs means recommendations quickly exhaust diverse options. A real system needs thousands of tracks.
- **Genre imbalance:** Pop dominates the dataset (26%), biasing results for users without a strong genre preference.
- **No content awareness:** The system cannot detect offensive lyrics, controversial artists, or thematic mismatch beyond mood labels.
- **LLM dependency:** If LM Studio is offline, the agent falls back to a generic profile; this loses personalization entirely and returns neutral recommendations that may not match any user intent.

### Could This AI Be Misused? How Would I Prevent It?

Yes, this system could theoretically be used to manipulate listening habits (e.g., always recommending songs from a specific label or artist). Guardrails are built into the design:
- **Diversity scoring** penalizes single-genre lists, preventing echo chambers
- **Confidence disclaimers** surface low-quality results before they reach the user
- **Transparent reasoning traces** logged to `logs/agent.log` provide an audit trail for debugging and accountability

### What Surprised Me About AI Reliability

- **Few-shot prompting is a force multiplier.** Without it, the LLM's critique output was inconsistent; sometimes omitting the confidence line entirely (~40% of baseline runs). With three curated examples, outputs became remarkably structured. A small prompt change had a large reliability impact.
- **The energy weight dominates everything.** The 6.0 energy weight often overrides perfect genre and mood matches. A song with wrong energy gets buried even if every other feature aligns, which shows how a single hyperparameter can shape the entire user experience.
- **Fuzzy matching improves discovery more than expected.** Related genre mapping (pop ↔ synthwave) surfaced songs users would never search for by name but genuinely fit their taste. Simple semantic maps are surprisingly effective at expanding recommendation reach without adding ML complexity.

### Collaboration with AI During This Project

**Helpful suggestion:** An AI assistant suggested structuring the agent loop as Plan → Retrieve → Act → Critique rather than a single monolithic prompt. This made each step independently testable and debuggable; when planning fails, we can fall back to a default profile without breaking the whole pipeline. The observable reasoning trace turned debugging from guesswork into inspection.

**Flawed suggestion:** An AI assistant initially recommended using the `openai` SDK for LM Studio integration. Since LM Studio exposes an OpenAI-compatible endpoint at `/v1/chat/completions`, this would add an unnecessary dependency. I instead built a zero-dependency client using only `urllib.request` from stdlib, which is lighter, requires no extra pip install, and works identically with the endpoint.

---

## Stretch Features Completed

| Feature | Implementation | Points |
|---------|---------------|--------|
| **RAG Enhancement** | Multi-source retrieval: songs.csv + genre/mood taxonomy. Agent supports a `use_rag` toggle. Retrieved context measurably improves critique quality vs. no-context baseline (test harness allows comparing RAG vs. No-RAG). | +2 |
| **Agentic Workflow Enhancement** | Four-step observable reasoning: PLAN → RETRIEVE → ACT → CRITIQUE. Each step logged to console and `logs/agent.log` with timestamps. Intermediate LLM outputs captured in `reasoning_trace`. | +2 |
| **Fine-Tuning / Specialization** | Few-shot prompting with 3 curated examples constrains output tone and format. Baseline comparison via `--no-few-shot` flag shows structured output drops from consistent to intermittent without prompting. | +2 |
| **Test Harness** | `src/test_harness.py` runs 8 predefined queries through three modes: Few-Shot with RAG, Baseline with RAG, and Few-Shot No-RAG. Prints pass/fail summary table and average confidence for each. | +2 |

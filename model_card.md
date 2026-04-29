# 🎧 Model Card: Rhymer (Agentic AI Edition)

## 1. Model Name  

**Rhymer** (Agentic AI Music Recommender)

---

## 2. Intended Use  

Rhymer is an AI-powered system designed to provide personalized music recommendations based on natural-language user queries.

- **Recommendations**: It generates ranked song lists with detailed explanations and confidence scores.
- **User Assumptions**: It assumes users can describe their musical preferences (genre, mood, activity) in plain English.
- **Purpose**: This is a classroom exploration project to demonstrate a hybrid AI architecture combining Large Language Models (LLMs) with deterministic scoring engines.

---

## 3. How the Model Works  

Rhymer uses a four-step **agentic workflow** to process user requests:

1.  **PLAN (LLM)**: An LLM extracts a structured "taste profile" (genre, mood, energy, tempo, etc.) from the user's natural-language query.
2.  **RETRIEVE (RAG)**: The system fetches relevant song data and genre/mood taxonomy maps using Retrieval-Augmented Generation (RAG) to provide context for the evaluation.
3.  **ACT (Deterministic Engine)**: A rule-based recommender scores songs from the catalog against the extracted profile using weighted genre, mood, and audio-feature matching.
4.  **CRITIQUE (LLM)**: The LLM evaluates the final recommendations, provides an explanation for the choices, and calculates a confidence score based on how well the results match the original intent.

The system uses **few-shot prompting** to ensure consistent output structure and **fuzzy matching** to identify related genres (e.g., matching "Pop" with "Synthwave").

---

## 4. Data  

- **Catalog**: A static dataset of 150 songs (`songs.csv`).
- **Context**: A taxonomy of genre and mood similarity maps used by the RAG retriever to understand semantic relationships.
- **Representation**: Includes genres like Pop, Metal, Lofi, Jazz, and EDM, and moods such as Chill, Intense, and Happy.
- **Missing Data**: Lacks real-time popularity data, lyrics, and deep sub-genre variety due to the small catalog size.

---

## 5. Strengths  

- **Natural Language Understanding**: Users can use conversational queries instead of rigid forms.
- **Explainability**: Every recommendation includes a point-by-point breakdown and a reasoning trace.
- **Reproducibility**: The deterministic scoring engine ensures that the same profile always yields the same rankings.
- **Quality Control**: Confidence disclaimers and LLM critiques catch and flag low-quality or vague results.

---

## 6. Limitations and Bias 

- **Small Catalog**: With 150 songs, the recommender has more variety than the original version, but still lacks the depth of major streaming platforms.
- **Genre Imbalance**: Pop dominates the dataset (26%), which can bias results toward mainstream tracks.
- **Hyperparameter Sensitivity**: The high weight on Energy (6.0) can override perfect genre or mood matches.
- **LLM Dependency**: The system requires a running LLM (e.g., via LM Studio); if offline, it falls back to a generic profile, losing personalization.
- **No Content Safety**: Does not detect offensive lyrics or controversial artists.

---

## 7. Evaluation  

- **Unit Testing**: 40 tests covering all modules (agent, evaluator, recommender, etc.).
- **Test Harness**: Validated across 8 diverse queries.
    - **Few-shot mode**: 7/8 passed (0.72 avg confidence).
    - **Baseline mode**: 5/8 passed (0.54 avg confidence).
- **Key Discovery**: Few-shot prompting reduced structured output failures from ~40% to 0%.

---

## 8. Future Work  

- **Catalog Expansion**: Scaling from 150 songs to thousands of tracks using a vector database.
- **Better Guardrails**: Improving handling of vague or out-of-domain queries.
- **Dataset Balancing**: Adding more entries for underrepresented genres like Flamenco or Grunge.
- **Multi-Modal Support**: Allowing users to provide sample songs or playlists as input.

---

## 9. Personal Reflection  

Building Rhymer taught me that **few-shot prompting is a force multiplier** for AI reliability. A small change in the prompt layout had a massive impact on the consistency of the critique output. 

I also discovered that hybrid architectures—combining the flexibility of LLMs with the predictability of math—are essential for building systems that are both powerful and explainable. Finding the balance between a "vibe" and numerical data (like energy vs. genre) remains a fascinating challenge in music recommendation.

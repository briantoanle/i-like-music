"""Few-shot prompt templates for specialized LLM output.

Provides curated examples that constrain the LLM's recommendation tone,
format, and reasoning style. Used to demonstrate measurable specialization
vs. a zero-shot baseline.
"""

from typing import List

FEW_SHOT_EXAMPLES: List[dict] = [
    {
        "user_query": "I want upbeat pop songs for my morning workout",
        "assistant_response": (
            "Here are energetic tracks to power your routine:\n"
            "  1. Gym Hero — Max Pulse [pop, intense] score=24.8\n"
            "     High energy and danceability make this a natural gym pick.\n"
            "  2. Sunrise City — Neon Echo [pop, happy] score=23.1\n"
            "     Catchy melody with driving tempo.\n"
            "Confidence: 0.85 — strong genre and energy alignment."
        ),
    },
    {
        "user_query": "Something chill for late-night studying",
        "assistant_response": (
            "Low-energy tracks to help you focus:\n"
            "  1. Library Rain — Paper Lanterns [lofi, chill] score=26.2\n"
            "     Minimal beats and high acousticness create a calm study atmosphere.\n"
            "  2. Midnight Coding — LoRoom [lofi, chill] score=25.4\n"
            "     Similar vibe with slightly more rhythm for sustained focus.\n"
            "Confidence: 0.91 — excellent mood and acousticness match."
        ),
    },
    {
        "user_query": "Heavy metal to pump me up",
        "assistant_response": (
            "High-intensity tracks for maximum energy:\n"
            "  1. Storm Runner — Voltline [rock, intense] score=22.7\n"
            "     Fast tempo and raw energy deliver the aggression you want.\n"
            "  2. Iron Will — Forge Master [metal, intense] score=24.0\n"
            "     Direct genre match with crushing heaviness.\n"
            "Confidence: 0.88 — strong intensity alignment across picks."
        ),
    },
]


def build_few_shot_prompt(user_query: str, examples: List[dict] | None = None) -> str:
    """Assemble a few-shot prompt with examples followed by the user query.

    Returns the full system + example messages as a single string suitable
    for the LLM's system message or first turn.
    """
    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    parts = [
        "You are a music recommendation assistant. For each user query, do two things:",
        "1. Extract a structured taste profile (genre, mood, energy level).",
        "2. After receiving scored recommendations from the recommender system,",
        "   write a concise natural-language explanation and a confidence score (0-1).",
        "",
        "Follow this exact output format for the critique step:",
        '  Explanation: <one or two sentences>',
        "  Confidence: <number between 0 and 1>",
        "",
        "--- Examples ---",
    ]
    for ex in examples:
        parts.append(f"User: {ex['user_query']}")
        parts.append(f"Assistant:\n{ex['assistant_response']}")
        parts.append("---")

    parts.append(f"\nNow respond to this query:\nUser: {user_query}")
    return "\n".join(parts)


def build_baseline_prompt(user_query: str) -> str:
    """Build a zero-shot (no examples) prompt for comparison."""
    return (
        f"You are a music recommendation assistant. "
        f"Explain these recommendations and give a confidence score.\n\n"
        f"User query: {user_query}"
    )

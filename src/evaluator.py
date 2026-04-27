"""Evaluator and confidence scoring for the music recommender agent.

Measures recommendation quality via diversity, score spread, and
combined confidence metrics. Acts as a guardrail layer.
"""

from typing import Dict, List


def evaluate_diversity(recommendations: List[Dict]) -> float:
    """Return 0–1 genre diversity score for the recommendation list.

    1.0 = all different genres, 0.0 = single genre (or empty list).
    """
    if len(recommendations) <= 1:
        return 1.0
    genres = {s.get("genre", "").lower() for s in recommendations}
    # (unique - 1) / (n - 1) → 0.0 when all same, 1.0 when all different
    return (len(genres) - 1) / (len(recommendations) - 1)


def evaluate_score_spread(scores: List[float]) -> float:
    """Return 0–1 score spread metric.

    Higher values mean the ranking is decisive (clear top picks).
    Lower values mean scores are clustered (hard to differentiate).
    """
    if len(scores) <= 1:
        return 1.0
    max_s, min_s = max(scores), min(scores)
    if max_s == min_s:
        return 0.0
    return (max_s - min_s) / max(max_s, 1.0)


def compute_confidence(llm_rating: float, diversity: float, spread: float) -> Dict[str, float]:
    """Combine LLM self-rating, diversity, and score spread into a confidence dict.

    Weights: LLM rating 50%, diversity 25%, spread 25%.
    """
    overall = llm_rating * 0.5 + diversity * 0.25 + spread * 0.25
    return {
        "overall": round(overall, 3),
        "llm_rating": round(llm_rating, 3),
        "diversity": round(diversity, 3),
        "spread": round(spread, 3),
    }


def needs_disclaimer(confidence: Dict[str, float], threshold: float = 0.3) -> bool:
    """Return True if a quality disclaimer should be appended."""
    return confidence["overall"] < threshold

"""Automated test harness for the music recommender agent.

Runs predefined queries through the agent and prints a pass/fail summary
with confidence ratings. Demonstrates the Test Harness stretch feature (+2).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from llm_client import LMStudioClient, LMStudioError
from agent import MusicAgent


TEST_QUERIES = [
    {"query": "I want chill lofi beats for studying",
     "expected_genre_hint": "lofi", "min_confidence": 0.3},
    {"query": "Upbeat pop songs for my morning workout",
     "expected_genre_hint": "pop", "min_confidence": 0.3},
    {"query": "Heavy metal to pump me up before a lift",
     "expected_genre_hint": "metal", "min_confidence": 0.3},
    {"query": "Relaxed jazz for a coffee shop afternoon",
     "expected_genre_hint": "jazz", "min_confidence": 0.2},
    {"query": "Electronic dance music for a party",
     "expected_genre_hint": "edm", "min_confidence": 0.3},
    {"query": "Sad acoustic songs for a rainy day",
     "expected_genre_hint": "", "min_confidence": 0.2},
    {"query": "Something energetic and positive to brighten my commute",
     "expected_genre_hint": "", "min_confidence": 0.2},
    {"query": "Ambient sounds for deep focus work",
     "expected_genre_hint": "ambient", "min_confidence": 0.2},
]


def run_harness(use_few_shot: bool = True) -> None:
    """Execute all test queries and print a summary table."""
    label = "Few-Shot" if use_few_shot else "Baseline (zero-shot)"
    print(f"\n{'=' * 70}")
    print(f"  Test Harness — {label} Mode")
    print(f"{'=' * 70}\n")

    try:
        agent = MusicAgent(
            songs_path="data/songs.csv",
            llm_client=LMStudioClient(),
            use_few_shot=use_few_shot,
        )
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    results = []
    confidences = []

    for i, tc in enumerate(TEST_QUERIES, 1):
        query = tc["query"]
        min_conf = tc["min_confidence"]
        print(f"[{i}/{len(TEST_QUERIES)}] Query: {query}")

        try:
            result = agent.run(query, k=5)
            recs_ok = len(result.recommendations) > 0
            conf_ok = result.confidence["overall"] >= min_conf
            explain_ok = bool(result.explanations.strip())
            passed = recs_ok and conf_ok and explain_ok

            status = "PASS" if passed else "FAIL"
            print(f"      {status} — recs={len(result.recommendations)}, "
                  f"confidence={result.confidence['overall']:.2f}, "
                  f"explanation={'yes' if explain_ok else 'no'}")

            results.append(passed)
            confidences.append(result.confidence["overall"])

        except Exception as e:
            print(f"      FAIL — error: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print(f"\n{'=' * 70}")
    print(f"  Summary ({label}): {passed}/{total} tests passed")
    print(f"  Average confidence: {avg_conf:.2f}")
    print(f"{'=' * 70}\n")


def main() -> None:
    """Run both few-shot and baseline harnesses for comparison."""
    run_harness(use_few_shot=True)
    run_harness(use_few_shot=False)


if __name__ == "__main__":
    main()

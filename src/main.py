"""Command line runner for the Music Recommender System.

Two modes:
  --agent  — Full agentic workflow (LLM plans, acts, critiques)
  --classic — Original rule-based recommender (backward compatible)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from recommender import load_songs, recommend_songs
from user_profile import POP_ENTHUSIAST, CHILL_LOFI_LISTENER, METAL_HEAD


def classic_mode() -> None:
    """Original rule-based recommendation output."""
    songs = load_songs("data/songs.csv")

    profiles = [
        ("High-Energy Pop Enthusiast", POP_ENTHUSIAST),
        ("Chill Lofi Student", CHILL_LOFI_LISTENER),
        ("Metal Head", METAL_HEAD),
    ]

    for profile_name, user_prefs in profiles:
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print("\n" + "=" * 65)
        print(f"{f'RECOMMENDATIONS FOR: {profile_name.upper()}':^65}")
        print("=" * 65 + "\n")

        for i, rec in enumerate(recommendations, 1):
            song, score, explanation = rec
            title_line = f" {i}. {song['title']} — {song['artist']}"
            score_line = f"    Match Score: {score:.2f}"
            reasons = explanation.replace("Points earned for: ", "").replace(".", "")
            if "General recommendation" in reasons:
                reasons = "Fits your baseline preferences"
            print(title_line)
            print(score_line)
            print(f"    Why: {reasons}")
            print("-" * 65)

    print("\n  Happy listening!\n")


def agent_mode(query: str, use_few_shot: bool = True) -> None:
    """Run the full agentic workflow for a natural-language query."""
    from llm_client import LMStudioClient
    from agent import MusicAgent

    client = LMStudioClient()
    agent = MusicAgent(songs_path="data/songs.csv",
                       llm_client=client,
                       use_few_shot=use_few_shot)

    result = agent.run(query, k=5)

    print("\n" + "=" * 65)
    print(f"{f'AGENT RECOMMENDATIONS':^65}")
    print("=" * 65 + "\n")
    print(f"Query: {query}\n")

    for i, rec in enumerate(result.recommendations, 1):
        song, score, explanation = rec
        reasons = explanation.replace("Points earned for: ", "").replace(".", "")
        if "General recommendation" in reasons:
            reasons = "Fits your baseline preferences"
        print(f" {i}. {song['title']} — {song['artist']}")
        print(f"    [{song.get('genre', '?')}, {song.get('mood', '?')}]  score={score:.2f}")
        print(f"    Why: {reasons}")
        print("-" * 65)

    print(f"\nExplanation: {result.explanations}")
    print(f"\nConfidence: {result.confidence['overall']:.2f} "
          f"(LLM={result.confidence['llm_rating']:.2f}, "
          f"diversity={result.confidence['diversity']:.2f}, "
          f"spread={result.confidence['spread']:.2f})")

    if result.reasoning_trace:
        print("\n--- Reasoning Trace ---")
        for entry in result.reasoning_trace:
            print(f"  {entry[:120]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Recommender System")
    parser.add_argument("--agent", nargs="?", const=True, default=None,
                        help="Run agentic mode. Pass a query string or use --agent 'your query'.")
    parser.add_argument("--classic", action="store_true", help="Run classic rule-based mode.")
    parser.add_argument("--no-few-shot", action="store_true",
                        help="Disable few-shot prompting (baseline comparison).")

    args = parser.parse_args()

    if args.agent is not None:
        query = args.agent if isinstance(args.agent, str) and args.agent else \
            input("What kind of music are you in the mood for? ")
        agent_mode(query, use_few_shot=not args.no_few_shot)
    elif args.classic:
        classic_mode()
    else:
        # Default to classic mode for backward compatibility
        classic_mode()


if __name__ == "__main__":
    main()

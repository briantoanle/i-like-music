"""Tests for agent module (unit tests, no LLM required).

Mocks the LMStudioClient so tests run without a running server.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock
from agent import MusicAgent, AgentResult


def _mock_llm(plan_json: str = '{"favorite_genre":"lofi","favorite_mood":"chill",'
                                '"target_energy":0.35,"target_tempo":75,'
                                '"target_valence":0.5,"target_danceability":0.4,'
                                '"likes_acoustic":true,"target_acousticness":0.8}',
              critique_text: str = "Explanation: Great lofi picks.\nConfidence: 0.85"):
    """Return a patched LMStudioClient that returns deterministic responses."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = [plan_json, critique_text]
    return mock_client


def test_agent_returns_result():
    mock_client = _mock_llm()
    agent = MusicAgent(songs_path="data/songs.csv", llm_client=mock_client)
    result = agent.run("chill lofi for studying")

    assert isinstance(result, AgentResult)
    assert len(result.recommendations) == 5
    assert isinstance(result.explanations, str)
    assert "overall" in result.confidence


def test_agent_result_has_trace():
    mock_client = _mock_llm()
    agent = MusicAgent(songs_path="data/songs.csv", llm_client=mock_client)
    result = agent.run("test query")

    assert len(result.reasoning_trace) >= 2
    # Should have RETRIEVE and ACT entries at minimum
    trace_text = " ".join(result.reasoning_trace)
    assert "RETRIEVE" in trace_text or "ACT" in trace_text


def test_agent_fallback_on_llm_failure():
    mock_client = MagicMock()
    # First call (_plan) returns None → fallback to default profile
    # Second call (_critique) returns a valid critique so the loop completes
    mock_client.chat.side_effect = [None, "Explanation: Fallback used.\nConfidence: 0.5"]

    agent = MusicAgent(songs_path="data/songs.csv", llm_client=mock_client)
    result = agent.run("anything")

    # Should still return results using default profile
    assert len(result.recommendations) == 5


def test_parse_json_valid():
    parsed = MusicAgent._parse_json('{"key": "value"}')
    assert parsed == {"key": "value"}


def test_parse_json_with_surrounding_text():
    parsed = MusicAgent._parse_json("Here's the profile: {\"genre\": \"pop\"} done.")
    assert parsed is not None
    assert parsed.get("genre") == "pop"


def test_parse_critique_valid():
    explanation, confidence = MusicAgent._parse_critique(
        "Explanation: Good picks.\nConfidence: 0.75"
    )
    assert "Good picks" in explanation
    assert abs(confidence - 0.75) < 0.01


def test_parse_critique_missing_confidence():
    _, confidence = MusicAgent._parse_critique("Just some text without format")
    assert confidence == 0.5  # default fallback
def test_parse_json_greedy_fix():
    # Bug 4: Greedy extraction from trailing text with braces
    text = '{"genre": "pop"} and then a note: {genre} is nice.'
    parsed = MusicAgent._parse_json(text)
    assert parsed is not None
    assert parsed.get("genre") == "pop"
    assert "note" not in parsed

def test_parse_critique_markdown_format():
    # Bug 5: Fragile startswith parsing
    text = "**Explanation:** These are perfect.\n**Confidence:** 0.95"
    explanation, confidence = MusicAgent._parse_critique(text)
    assert "perfect" in explanation
    assert confidence == 0.95

def test_parse_critique_numbered_list():
    text = "1. Explanation: Very accurate.\n2. Confidence: 0.88"
    explanation, confidence = MusicAgent._parse_critique(text)
    assert "accurate" in explanation
    assert confidence == 0.88

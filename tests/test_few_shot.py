"""Tests for few-shot prompt module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from few_shot import FEW_SHOT_EXAMPLES, build_few_shot_prompt, build_baseline_prompt


def test_examples_non_empty():
    assert len(FEW_SHOT_EXAMPLES) >= 3


def test_example_structure():
    for ex in FEW_SHOT_EXAMPLES:
        assert "user_query" in ex
        assert "assistant_response" in ex
        assert isinstance(ex["user_query"], str)
        assert isinstance(ex["assistant_response"], str)


def test_few_shot_prompt_contains_examples():
    prompt = build_few_shot_prompt("test query")
    assert "Examples" in prompt
    assert "Confidence:" in prompt
    assert "test query" in prompt


def test_baseline_prompt_no_examples():
    prompt = build_baseline_prompt("test query")
    assert "test query" in prompt
    # Baseline should be shorter than few-shot
    assert len(prompt) < len(build_few_shot_prompt("test query"))


def test_custom_examples_used():
    custom = [{"user_query": "custom", "assistant_response": "response"}]
    prompt = build_few_shot_prompt("query", examples=custom)
    assert "custom" in prompt
    # Default examples should not appear when custom ones are passed
    for ex in FEW_SHOT_EXAMPLES:
        assert ex["user_query"] not in prompt

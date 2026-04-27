"""Tests for evaluator module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluator import evaluate_diversity, evaluate_score_spread, compute_confidence, needs_disclaimer


def test_diversity_all_same_genre():
    recs = [{"genre": "pop"} for _ in range(5)]
    assert evaluate_diversity(recs) == 0.0


def test_diversity_all_different():
    recs = [{"genre": g} for g in ["pop", "rock", "lofi", "jazz", "edm"]]
    assert evaluate_diversity(recs) == 1.0


def test_diversity_single_song():
    assert evaluate_diversity([{"genre": "pop"}]) == 1.0


def test_diversity_empty():
    assert evaluate_diversity([]) == 1.0


def test_score_spread_identical_scores():
    assert evaluate_score_spread([5.0, 5.0, 5.0]) == 0.0


def test_score_spread_wide_range():
    spread = evaluate_score_spread([10.0, 20.0, 30.0])
    assert 0.0 < spread <= 1.0


def test_confidence_combination():
    conf = compute_confidence(llm_rating=0.8, diversity=1.0, spread=0.5)
    # 0.8*0.5 + 1.0*0.25 + 0.5*0.25 = 0.4 + 0.25 + 0.125 = 0.775
    assert abs(conf["overall"] - 0.775) < 0.001


def test_disclaimer_needed():
    assert needs_disclaimer({"overall": 0.2}) is True
    assert needs_disclaimer({"overall": 0.5}) is False

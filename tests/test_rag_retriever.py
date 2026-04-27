"""Tests for RAG retriever module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_retriever import retrieve_context, build_prompt_context, get_taxonomy_context, _expand_keywords, GENRE_RELATED


def test_retrieve_context_returns_string():
    songs = {
        1: {"title": "Test Song", "artist": "Artist", "genre": "pop",
            "mood": "happy", "energy": 0.8, "tempo_bpm": 120},
    }
    ctx = retrieve_context(songs)
    assert isinstance(ctx, str)
    assert "Test Song" in ctx


def test_retrieve_context_filters_by_genre():
    songs = {
        i: {"title": f"Song{i}", "artist": "A", "genre": "lofi" if i < 5 else "pop",
             "mood": "chill" if i < 5 else "happy", "energy": 0.4, "tempo_bpm": 80}
        for i in range(1, 11)
    }
    ctx = retrieve_context(songs, query_hint="lofi study beats", top_n=6)
    # Should include lofi songs first
    assert "Song1" in ctx


def test_build_prompt_context_formats_recommendations():
    songs = {1: {"title": "A", "artist": "B", "genre": "pop", "mood": "happy"}}
    recs = [(songs[1], 25.0, "genre match")]
    ctx = build_prompt_context(recs, songs)
    assert "A" in ctx
    assert "B" in ctx
    assert "score=25.0" in ctx


def test_taxonomy_context_non_empty():
    tax = get_taxonomy_context()
    assert "Genre relationships" in tax
    assert "Mood relationships" in tax
    assert len(tax) > 100


def test_expand_keywords_includes_neighbors():
    result = _expand_keywords("pop", GENRE_RELATED)
    assert "pop" in result
    assert "edm" in result
    assert "synthwave" in result

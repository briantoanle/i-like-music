"""RAG retriever for the music recommender.

Extracts song metadata from the CSV catalog and formats it as context
for LLM prompts. Supports keyword filtering and genre/mood taxonomy.
"""

from typing import Dict, List, Optional


# Shared genre/mood adjacency — mirrors the mappings in recommender.py
GENRE_RELATED = {
    "pop": ["synthwave", "disco", "edm", "indie pop", "r&b"],
    "rock": ["metal", "punk", "grunge", "psychedelic rock"],
    "lofi": ["ambient", "chill", "jazz", "acoustic"],
    "electronic": ["edm", "techno", "synthwave", "ambient"],
}

MOOD_RELATED = {
    "chill": ["relaxed", "peaceful", "focused", "smooth", "minimalist"],
    "happy": ["upbeat", "sunny", "party", "catchy", "hopeful"],
    "intense": ["angry", "aggressive", "energetic", "passionate"],
    "sad": ["melancholy", "emotional", "gloomy"],
}


def _expand_keywords(keyword: str, mapping: Dict[str, List[str]]) -> List[str]:
    """Return a list containing the taxonomical key plus all related neighbours if a match is found."""
    kw = keyword.lower().strip()
    expanded = set()
    for key, neighbors in mapping.items():
        # Exact match or neighbor match only (avoids substring matching for expansion)
        if kw == key or kw in neighbors:
            expanded.add(key)
            expanded.update(neighbors)
    return list(expanded)


def retrieve_context(songs: Dict[int, Dict], query_hint: str = "", top_n: int = 10,
                     max_chars: int = 1500) -> str:
    """Return formatted song metadata as a text block for LLM context.

    If *query_hint* contains genre or mood words, filter the catalog first,
    then pad with remaining songs up to *top_n*. Stops formatting once
    *max_chars* is reached to avoid generating oversized prompts.
    """
    query_lower = query_hint.lower()

    # Filter out short stopwords and punctuation to avoid pollution (Bug 3)
    query_words = [w.strip("?!.,()[]'\"").lower() for w in query_lower.split()]
    query_words = [w for w in query_words if len(w) > 2]

    # Collect candidate keywords from the query
    genre_keywords: List[str] = []
    mood_keywords: List[str] = []
    
    for kw in query_words:
        # Check all categories for every word to avoid blind spots (Bug 2)
        g_expanded = _expand_keywords(kw, GENRE_RELATED)
        m_expanded = _expand_keywords(kw, MOOD_RELATED)
        
        if g_expanded:
            genre_keywords.extend(g_expanded)
        if m_expanded:
            mood_keywords.extend(m_expanded)
            
        # If no expansion found, still treat the word as a potential literal keyword
        # but only if it's not a common stopword (handled by length check above)
        if not g_expanded and not m_expanded:
            genre_keywords.append(kw)
            mood_keywords.append(kw)

    genre_set = set(g.lower() for g in genre_keywords)
    mood_set = set(m.lower() for m in mood_keywords)

    matched: List[Dict] = []
    rest: List[Dict] = []
    if genre_set or mood_set:
        for song in songs.values():
            s_genre = song.get("genre", "").lower()
            s_mood = song.get("mood", "").lower()
            if (genre_set and s_genre in genre_set) or (mood_set and s_mood in mood_set):
                matched.append(song)
            else:
                rest.append(song)
    else:
        rest = list(songs.values())

    # Matched first, then pad with remaining — up to top_n
    candidates = matched + rest[:max(0, top_n - len(matched))]

    return _format_songs(candidates, max_chars=max_chars)


def build_prompt_context(recommendations: List[tuple], songs: Dict[int, Dict]) -> str:
    """Format the recommender's top-K results as a concise context block.

    Each *recommendation* is ``(song_dict, score, explanation)``.
    """
    lines = []
    for song, score, _ in recommendations:
        sid = song.get("id", "?")
        if sid in songs:
            s = songs[sid]
        else:
            s = song
        lines.append(
            f"  • {s.get('title', '?')} by {s.get('artist', '?')} "
            f"[{s.get('genre', '?')}, {s.get('mood', '?')}, {s.get('release_year', '?')}] "
            f"score={score}"
        )
    return "\n".join(lines)


def _format_songs(songs: List[Dict], max_chars: int = 1500) -> str:
    """Format a list of song dicts as a compact table-like string.

    Stops adding songs once *max_chars* is reached to avoid oversized prompts.
    """
    header = "Song catalog context (title | artist | genre | mood | energy | tempo | year):"
    lines = [header]
    budget = max_chars - len(header) - 1
    for s in songs:
        line = (
            f"\n  {s.get('title', '?')} | {s.get('artist', '?')} | "
            f"{s.get('genre', '?')} | {s.get('mood', '?')} | "
            f"E={s.get('energy', '?')} T={s.get('tempo_bpm', '?')} Y={s.get('release_year', '?')}"
        )
        if len(line) > budget:
            break
        lines.append(line)
        budget -= len(line)
    return "".join(lines)


def get_taxonomy_context() -> str:
    """Return the genre and mood taxonomy as text for multi-source RAG."""
    parts = ["Genre relationships:", _format_mapping(GENRE_RELATED)]
    parts.append("Mood relationships:")
    parts.append(_format_mapping(MOOD_RELATED))
    return "\n".join(parts)


def _format_mapping(mapping: Dict[str, List[str]]) -> str:
    lines = []
    for key, neighbors in mapping.items():
        lines.append(f"  {key} ↔ {', '.join(neighbors)}")
    return "\n".join(lines)

import heapq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    release_year: int = 2024

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_tempo: float
    target_valence: float
    target_danceability: float
    likes_acoustic: bool
    target_acousticness: float

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Initializes the recommender with a list of Song objects."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Generates top-k song recommendations for a given user profile."""
        # Convert dataclass to dict to reuse functional score_song logic
        user_dict = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "target_tempo": user.target_tempo,
            "target_valence": user.target_valence,
            "target_danceability": user.target_danceability,
            "likes_acoustic": user.likes_acoustic,
            "target_acousticness": user.target_acousticness
        }

        # Pythonic Top-K calculation using stable sorting: 
        # Sort by score DESC, then artist ASC, then title ASC
        scored = sorted(
            [(s, score_song(user_dict, vars(s))[0]) for s in self.songs],
            key=lambda x: (-x[1], x[0].artist, x[0].title)
        )
        
        return [s for s, score in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explains the recommendation score for a specific song and user."""
        user_dict = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "target_tempo": user.target_tempo,
            "target_valence": user.target_valence,
            "target_danceability": user.target_danceability,
            "likes_acoustic": user.likes_acoustic,
            "target_acousticness": user.target_acousticness
        }
        s_dict = vars(song)
        _, explanation = score_song(user_dict, s_dict)
        return explanation

import csv

def get_genre_similarity(u_genre: str, s_genre: str) -> float:
    """Returns a similarity coefficient (0.0 to 1.0) between two genres."""
    u = u_genre.lower().strip()
    s = s_genre.lower().strip()
    if not u or not s: return 0.0
    if u == s: return 1.0
    # Only allow substring matches for words longer than 2 chars (Bug 3)
    if (len(u) > 2 and u in s) or (len(s) > 2 and s in u): return 0.7
    
    # Simple semantic mapping
    related = {
        "pop": ["synthwave", "disco", "edm", "indie pop", "r&b"],
        "rock": ["metal", "punk", "grunge", "psychedelic rock"],
        "lofi": ["ambient", "chill", "jazz", "acoustic"],
        "electronic": ["edm", "techno", "synthwave", "ambient"],
    }
    for key, neighbors in related.items():
        if (u == key and s in neighbors) or (s == key and u in neighbors):
            return 0.5
    return 0.0

def get_mood_similarity(u_mood: str, s_mood: str) -> float:
    """Returns a similarity coefficient (0.0 to 1.0) between two moods."""
    u = u_mood.lower().strip()
    s = s_mood.lower().strip()
    if not u or not s: return 0.0
    if u == s: return 1.0
    
    related = {
        "chill": ["relaxed", "peaceful", "focused", "smooth", "minimalist"],
        "happy": ["upbeat", "sunny", "party", "catchy", "hopeful"],
        "intense": ["angry", "aggressive", "energetic", "intense", "passionate"],
        "sad": ["melancholy", "emotional", "gloomy"],
    }
    for key, neighbors in related.items():
        if (u == key and s in neighbors) or (s == key and u in neighbors):
            return 0.6
    return 0.0

def load_songs(csv_path: str) -> Dict[int, Dict]:
    """Loads songs from a CSV file into a dictionary keyed by song ID."""
    songs = {}
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numerical strings to floats/ints
                song = {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                    "release_year": int(row["release_year"])
                }
                songs[song["id"]] = song
    except FileNotFoundError:
        print(f"Error: Could not find songs file at {csv_path}")
    except Exception as e:
        print(f"Error loading songs: {e}")
    
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, str]:
    """Calculates a numerical match score and explanation for a song and user profile."""
    score = 0.0
    reasons = []

    # Rebalanced Weights (Prioritizing "vibe" over strict "labels" to break filter bubbles)
    W_GENRE = 3.5
    W_MOOD = 2.5
    W_ENERGY = 6.0
    W_TEMPO = 4.0
    W_ACOUSTIC = 4.0
    W_VALENCE = 3.0
    W_DANCE = 3.0

    # 1. Fuzzy Genre match
    u_genre = user_prefs.get("favorite_genre", "")
    s_genre = song.get("genre", "")
    genre_sim = get_genre_similarity(u_genre, s_genre)
    if genre_sim > 0:
        score += W_GENRE * genre_sim
        if genre_sim == 1.0:
            reasons.append(f"genre match ({s_genre})")
        else:
            reasons.append(f"related genre ({s_genre})")

    # 2. Fuzzy Mood match
    u_mood = user_prefs.get("favorite_mood", "")
    s_mood = song.get("mood", "")
    mood_sim = get_mood_similarity(u_mood, s_mood)
    if mood_sim > 0:
        score += W_MOOD * mood_sim
        if mood_sim == 1.0:
            reasons.append(f"mood match ({s_mood})")
        else:
            reasons.append(f"similar mood ({s_mood})")

    # 3. Energy similarity (Loosened thresholds to reduce "invisible" logic)
    target_energy = user_prefs.get("target_energy", 0.5)
    energy_diff = abs(song["energy"] - target_energy)
    score += max(0, W_ENERGY * (1 - energy_diff))
    if energy_diff < 0.15:
        reasons.append("perfect energy")
    elif energy_diff < 0.4:
        reasons.append("good energy profile")

    # 4. Acousticness
    target_acoustic = user_prefs.get("target_acousticness", 0.5)
    acoustic_diff = abs(song["acousticness"] - target_acoustic)
    score += max(0, W_ACOUSTIC * (1 - acoustic_diff))
    if acoustic_diff < 0.25:
        reasons.append("ideal acousticness")

    # 5. Tempo similarity (Loosened threshold)
    target_tempo = user_prefs.get("target_tempo", 120)
    tempo_diff = abs(song["tempo_bpm"] - target_tempo)
    tempo_score = max(0, W_TEMPO * (1 - (tempo_diff / 100)))
    score += tempo_score
    if tempo_diff < 25:
        reasons.append("matching tempo")

    # 6. Valence & Danceability
    v_diff = abs(song["valence"] - user_prefs.get("target_valence", 0.5))
    d_diff = abs(song["danceability"] - user_prefs.get("target_danceability", 0.5))
    score += max(0, W_VALENCE * (1 - v_diff))
    score += max(0, W_DANCE * (1 - d_diff))
    if v_diff < 0.2 and d_diff < 0.2:
        reasons.append("perfect vibe")

    explanation = "Points earned for: " + ", ".join(reasons) + "." if reasons else "General recommendation."
    return round(score, 2), explanation

def recommend_songs(user_prefs: Dict, songs: Dict[int, Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Generates top-k song recommendations using a functional approach with secondary tie-breaking."""
    scored_songs = []
    for song in songs.values():
        score, explanation = score_song(user_prefs, song)
        # Use (score, artist, title) for stable secondary sorting
        scored_songs.append((song, score, explanation))

    # Second pass for stable sorting: Sort by score DESC, then artist ASC, then title ASC
    scored_songs = sorted(
        scored_songs, 
        key=lambda x: (-x[1], x[0]["artist"], x[0]["title"])
    )
    
    return scored_songs[:k]

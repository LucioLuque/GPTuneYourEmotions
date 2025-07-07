import numpy as np
import random
from typing import List
from emotions.emotions import data_ids, lyrics_by_id, emotional_data_embeddings
from context.context import contextual_data_embeddings
from sklearn.metrics.pairwise import cosine_similarity

def get_k_mid_points(emotional_embedding_1: np.ndarray, emotional_embedding_2: np.ndarray, k: int = 2) -> List[np.ndarray]:
    """
    Returns the k mid points of the emotional embedding.
    --------
    Args:
        emotional_embedding_1 (np.ndarray): The first emotional embedding.
        emotional_embedding_2 (np.ndarray): The second emotional embedding.
        k (int): The number of mid points to return. Default is 2.
    --------
    Returns:
        List[np.ndarray]: A list of k mid points between the two emotional embeddings.
    """
    mid_points= []
    t = emotional_embedding_2 - emotional_embedding_1
    for i in range(k):
        mid_point = emotional_embedding_1 + ((i)/(k-1))* t
        mid_points.append(mid_point)
    return mid_points

def choose_ids_weighted(emo_sims: np.ndarray, ctx_sims: np.ndarray, k: int, w_emo: float,
                        w_ctx: float, selection: str = 'best', n: int = 1):
    """
    For each of the k mid-points, calculates the combined similarity score
    and selects the n songs with the highest score (or a random sample
    within the top-n if selection='random').
    --------
    Args:
        emo_sims (np.ndarray): Emotional similarity scores for k mid-points.
        ctx_sims (np.ndarray): Contextual similarity scores for k mid-points.
        k (int): Number of mid-points.
        w_emo (float): Weight for emotional similarity.
        w_ctx (float): Weight for contextual similarity.
        selection (str): Selection mode, either 'best' or 'random'.
        n (int): Number of songs to select per mid-point.
    --------
    Returns:
        list: A list of selected song IDs.
    """
    chosen = []
    used = set()

    MIN_WORDS = 50

    for i in range(k):
        
        combined = w_emo * emo_sims[i] + w_ctx * ctx_sims[i] # combined score

        sorted_idxs = np.argsort(combined)[::-1]

        if used:
            sorted_idxs = [idx for idx in sorted_idxs if idx not in used]

        sorted_idxs = [ idx for idx in sorted_idxs if data_ids[idx] in lyrics_by_id and
            isinstance(lyrics_by_id[data_ids[idx]], str) and len(lyrics_by_id[data_ids[idx]].strip().split()) >= MIN_WORDS]

        if selection == 'best':
            pick = sorted_idxs[:n]
        elif selection == 'random':
            top_pool = sorted_idxs[: n*2 ]
            if len(top_pool) < n:
                raise ValueError(f"Pocos candidatos: {len(top_pool)} disponibles, {n} requeridos.")
            pick = random.sample(top_pool, n)
        else:
            raise ValueError("selection debe ser 'best' o 'random'.")

        chosen.extend(pick)
        used.update(pick)

    return chosen

def get_playlist_ids_weighted(emotional_emb1: np.ndarray, emotional_emb2: np.ndarray, contextual_emb1: np.ndarray, 
                               contextual_emb2: np.ndarray, k: int = 2, weight_emotion: float = 0.5, 
                               weight_context: float = 0.5, selection: str = 'best', n: int = 1) -> List[str]:
    """
    Selects k songs based on emotional and contextual embeddings.
    --------
    Args:
        emotional_emb1 (np.ndarray): First emotional embedding.
        emotional_emb2 (np.ndarray): Second emotional embedding.
        contextual_emb1 (np.ndarray): First contextual embedding.
        contextual_emb2 (np.ndarray): Second contextual embedding.
        k (int): Number of mid points to consider. Default is 2.
        weight_emotion (float): Weight for emotional similarity. Default is 0.5.
        weight_context (float): Weight for contextual similarity. Default is 0.5.
        selection (str): Selection mode, either 'best' or 'random'. Default is 'best'.
        n (int): Number of songs to select per mid-point. Default is 1.
    --------
    Returns:
        list: A list of selected song IDs.
    """
    if not isinstance(emotional_emb1, np.ndarray):
        emotional_emb1 = np.array(emotional_emb1)
    if not isinstance(emotional_emb2, np.ndarray):
        emotional_emb2 = np.array(emotional_emb2)
    if not isinstance(contextual_emb1, np.ndarray):
        contextual_emb1 = np.array(contextual_emb1)
    if not isinstance(contextual_emb2, np.ndarray):
        contextual_emb2 = np.array(contextual_emb2)
    # Get emotional and contextual mid points
    emo_k = get_k_mid_points(emotional_emb1, emotional_emb2, k)
    ctx_k = get_k_mid_points(contextual_emb1, contextual_emb2, k)
    # Similarity calculations
    emo_sims = cosine_similarity(emo_k, emotional_data_embeddings)
    ctx_sims = cosine_similarity(ctx_k,   contextual_data_embeddings)
    # Weighted selection
    chosen_idxs = choose_ids_weighted(emo_sims, ctx_sims, k,
                                      weight_emotion, weight_context,
                                      selection, n)
    return [data_ids[i] for i in chosen_idxs]
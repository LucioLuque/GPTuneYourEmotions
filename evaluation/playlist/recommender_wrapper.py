# evaluation/recommender_wrapper.py
from emotions.emotions import detect_user_emotions, get_playlist_ids2_weighted
from context.context import get_context_embedding 

def recommend_from_prompts(prompt1: str, prompt2: str, ):
    """Convierte los dos mensajes en embeddings y devuelve los IDs de la playlist."""
    emb1, _ = detect_user_emotions(prompt1, n=3)
    emb2, _ = detect_user_emotions(prompt2, n=3)
    context_embedding_1 = get_context_embedding(prompt1)  
    context_embedding_2 = get_context_embedding(prompt2)  
    # usa mismos parámetros que app.py
    return get_playlist_ids2_weighted(
        emb1, emb2,
        context_embedding_1, context_embedding_2,
        genres=[],              # sin filtros
        k=5, selection='best',
        mode='interpolation', n=4     # → 11 canciones
    )

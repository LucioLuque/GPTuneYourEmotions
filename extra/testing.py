# from model import generate_reflection, generate_recommendation
from GPTuneYourEmotions.emotions.emotions import detect_user_emotions, get_k_mid_points, emotions_labels, filtered_labels, format_top_emotions, get_playlist_ids, data_embeddings, data_ids
# from song import get_song
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

msg_1 = "I feel so lost and helpless right now, like nothing is going right."
msg_2 = "I wish I could just feel happy again, like I used to."
emotional_embedding_1, top_emotions_detected_1 = detect_user_emotions(msg_1, n=3)
emotional_embedding_2, top_emotions_detected_2 = detect_user_emotions(msg_2, n=3)
# print(f"Emotional embedding 1: {emotional_embedding_1}")
# print(f"Top emotions detected 1: {top_emotions_detected_1}")

import random

def choose_ids(similarities, k, m=0):
    chosen_ids = []            
    used_indices = set()       

    for i in range(k):
        sim_vector = similarities[i]      
        
        # Sort the similarity vector in descending order
        sorted_indices = np.argsort(sim_vector)[::-1] 
        
        # m closest indices
        top_m = sorted_indices[:m]

        available = [idx for idx in top_m if idx not in used_indices]

        if available:
            # Si queda al menos uno libre dentro de los m más cercanos, elijo al azar uno de ellos
            chosen_idx = random.choice(available)
        else:
            #si todos los m más cercanos ya se usan, recorro sorted_indices hasta
            # encontrar el primero que no esté en used_indices
            chosen_idx = None
            for idx in sorted_indices:
                if idx not in used_indices:
                    chosen_idx = idx
                    break
            # por las dudas, si no se encuentra ninguno, lanzo un error
            if chosen_idx is None:
                raise ValueError(
                    f"No quedan embeddings disponibles para el punto medio #{i}."
                )

        used_indices.add(chosen_idx)
        chosen_ids.append(chosen_idx)

    return chosen_ids

def get_playlist_ids_2(embedding_1, embedding_2, genres=[], k=2, m=1):
    """
    Returns the IDs of the playlist songs that are closest to the k mid points between 
    two emotional embeddings.
    Default k=2 value returns the closest songs to the original inputs.
    """
    #if embeddings are not numpy arrays, convert them
    if not isinstance(embedding_1, np.ndarray):
        embedding_1 = np.array(embedding_1)
    if not isinstance(embedding_2, np.ndarray):
        embedding_2 = np.array(embedding_2)
    mid_points = get_k_mid_points(embedding_1, embedding_2, k)
    similarities = cosine_similarity(mid_points, data_embeddings)
    closest_idxs = choose_ids(similarities, k, m)
    print(len(closest_idxs))
    closest_ids = [data_ids[idx] for idx in closest_idxs]
    return closest_ids

songs_ids = get_playlist_ids_2(emotional_embedding_1, emotional_embedding_2, None, k=5, m=10)
songs_ids_2 = get_playlist_ids(emotional_embedding_1, emotional_embedding_2, None, k=5)
print(songs_ids)
print(songs_ids_2)

# mid_points = get_k_mid_points(emotional_embedding_1, emotional_embedding_2, k=5)
# for i, mid_point in enumerate(mid_points):
#     dict_emotions = {label: mid_point[idx] for idx, label in enumerate(emotions_labels)}
#     if i==0:
#         print(f"Actual emotions: {dict_emotions}")
         
       
#     if i==len(mid_points)-1:
#         print(f"Desired emotions: {dict_emotions}")
#     else:
#         print(f"Mid point {i+1}: {dict_emotions}")
#     top_emotions = format_top_emotions(mid_point, filtered_labels, top_n=3)
#     print(f"Top emotions at mid point {i+1}: {top_emotions}")


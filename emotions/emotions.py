from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import no_grad
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTIONAL_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "all_embeddings.npy")
DATASET_PATH = os.path.join(ROOT_DIR, "data", "dataset_with_embeddings.csv")
CONTEXTUAL_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "all_context_embeddings.npy")

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model     = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
model.eval()

id2label = model.config.id2label

neutral_idx = next(i for i,lbl in id2label.items() if lbl == "neutral")
_mask = np.arange(len(id2label)) != neutral_idx

emotional_data_embeddings = np.load(EMOTIONAL_EMBEDDINGS_PATH)
df = pd.read_csv(DATASET_PATH)
data_ids = df["id"]
lyrics_by_id = dict(zip(df["id"], df["lyrics"]))  #agregado


filtered_labels = {
    new_i: id2label[old_i]
    for new_i, old_i in enumerate(np.where(_mask)[0])
}

emotions_labels = list(filtered_labels.values())

contextual_data_embeddings = np.load(CONTEXTUAL_EMBEDDINGS_PATH)

def get_token_chunks(text, tokenizer, max_length=512, stride=384):
    encoding = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encoding["input_ids"][0]
    chunks = [] 
    for start in range(0, len(input_ids), stride):
        end = start + max_length
        chunk = input_ids[start:end]
        chunks.append(chunk)
        if end >= len(input_ids):
            break

    return chunks

def format_top_emotions(probs, labels, top_n=1):
    """
    Formats the top_k emotions from the probabilities
    """
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]
    return [labels[i] for i in top_indices]

def get_emotional_embedding_v1(text, model, tokenizer):
    """ 
    Computes the emotional embedding of a given text by extracting averagint the 
    logits of the model for each chunk of the text.
    """
    token_chunks = get_token_chunks(text, tokenizer)
    
    chunk_logits = []
    for chunk in token_chunks:
        with no_grad():
            outputs = model(chunk.unsqueeze(0))
            logits = outputs.logits
            chunk_logits.append(logits.squeeze().numpy())
    
    avg_logits = np.mean(chunk_logits, axis=0)
    return avg_logits

def take_out_neutral_emotion(emotional_embedding):
    """
    Takes out the neutral emotion from the emotional embedding.
    """
    return emotional_embedding[_mask], filtered_labels

# def detect_user_emotion(user_input):
#     emotional_embedding_1 = get_emotional_embedding_v1(user_input, model, tokenizer)
#     emotional_embedding_2, labels_2 = take_out_neutral_emotion(emotional_embedding_1)
#     top_emotion = format_top_emotions(emotional_embedding_2, labels_2, top_k=1)[0][0]
#     return emotional_embedding_2, top_emotion

def detect_user_emotions(user_input, n=1):
    emotional_embedding_1 = get_emotional_embedding_v1(user_input, model, tokenizer)
    emotional_embedding_2, labels_2 = take_out_neutral_emotion(emotional_embedding_1)
    top_emotion = format_top_emotions(emotional_embedding_2, labels_2, top_n=n)
    return emotional_embedding_2, top_emotion

def get_k_mid_points(emotional_embedding_1, emotional_embedding_2, k=2):
    """
    Returns the k mid points of the emotional embedding.
    """
    mid_points= []
    t = emotional_embedding_2 - emotional_embedding_1
    for i in range(k):
        mid_point = emotional_embedding_1 + ((i)/(k-1))* t
        mid_points.append(mid_point)
    return mid_points

def get_available_indexes(sorted_indexes, used_indexes, m):
    """
    Returns a list of available indexes from sorted_indexes that are not in used_indexes,
    limited to the first m elements.
    """
    #hacer funcion de chequeos
    available = []
    for idx in sorted_indexes:
        if idx not in used_indexes:
            available.append(idx)
            if len(available) == m:
                break
    return available

def choose_ids(similarities, k, selection='best', m=1, 
               mode='constant', n=1):
    #hacer funcion de chequeos
    chosen_ids = []            
    used_indexes = set()
    if mode == 'constant':
        #constant number of songs (n) to select for each mid point	
        n_to_selects = [n] * k
    elif mode == 'interpolation':
        if n == 0:
            #interpolacion lineal entre 1 y k
            n_to_selects = [i + 1 for i in range(k)]
        else:
            #interpolacion lineal entre 1 y n
            n_to_selects = np.linspace(1, n, k, dtype=int).tolist() 
    else:
        raise ValueError("Invalid mode. Use 'constant' or 'interpolation'.")

    print(f"Number of songs to select for each mid point: {n_to_selects}")
    for i in range(k):
        sim_vector = similarities[i]

        sorted_indexes = np.argsort(sim_vector)[::-1]
        n_to_select = n_to_selects[i]
        dynamic_m = m
        if dynamic_m <= n_to_select:
            print(f"Had to increase m from {dynamic_m} to {n_to_select*2} to select enough songs.")
            dynamic_m = n_to_select*2
        
        available = get_available_indexes(sorted_indexes, used_indexes, dynamic_m)
        if selection == 'best':
            chosen_idxs = available[:n_to_select]
        elif selection == 'random':
            chosen_idxs = random.sample(available, n_to_select)
        
        chosen_ids.extend(chosen_idxs)
        used_indexes.update(chosen_idxs)
    return chosen_ids

def get_playlist_ids(embedding_1, embedding_2, genres=[], k=2, selection = 'best', m=1, mode='constant', n=1):
                 
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
    #hacer funcion de chequeos
    mid_points = get_k_mid_points(embedding_1, embedding_2, k)
    similarities = cosine_similarity(mid_points, emotional_data_embeddings)
    closest_idxs = choose_ids(similarities, k, selection, m, mode, n)
    print(f"Amount of songs selected: {len(closest_idxs)}")
    closest_ids = [data_ids[idx] for idx in closest_idxs]
    return closest_ids

#filtrar por emociones y luego por contexto
def get_playlist_ids2(emotional_emb1, emotional_emb2, contextual_emb1, contextual_emb2,  genres=[], k=2, selection = 'best', m=1, mode='constant', n=1):
                 
    """
    Returns the IDs of the playlist songs that are closest to the k mid points between 
    two emotional embeddings.
    Default k=2 value returns the closest songs to the original inputs.
    """
    #if embeddings are not numpy arrays, convert them
    if not isinstance(emotional_emb1, np.ndarray):
        emotional_emb1 = np.array(emotional_emb1)
    if not isinstance(emotional_emb2, np.ndarray):
        emotional_emb2 = np.array(emotional_emb2)
    if not isinstance(contextual_emb1, np.ndarray):
        contextual_emb1 = np.array(contextual_emb1)
    if not isinstance(contextual_emb2, np.ndarray):
        contextual_emb2 = np.array(contextual_emb2)

    #hacer funcion de chequeos
    emotional_mid_points = get_k_mid_points(emotional_emb1, emotional_emb2, k)
    emotional_similarities = cosine_similarity(emotional_mid_points, emotional_data_embeddings)  #poner emotional_data_emb
    contextual_mid_points = get_k_mid_points(contextual_emb1, contextual_emb2, k)

    closest_idxs = choose_ids_dual_filter(emotional_similarities, contextual_mid_points, k,
                                          selection=selection, m=m, mode=mode, n=n)
    
    print(f"Amount of songs selected: {len(closest_idxs)}")
    closest_ids = [data_ids[idx] for idx in closest_idxs]
    return closest_ids

def choose_ids_dual_filter(emotional_sims, contextual_midpoints, k, selection='best', m=1, mode='constant', n=1):
    chosen_ids = []
    used_indexes = set()

    # Determinar cuántas canciones seleccionar por punto
    if mode == 'constant':
        n_to_selects = [n] * k
    elif mode == 'interpolation':
        if n == 0:
            n_to_selects = [i + 1 for i in range(k)]
        else:
            n_to_selects = np.linspace(1, n, k, dtype=int).tolist()
    else:
        raise ValueError("Invalid mode. Use 'constant' or 'interpolation'.")

    print(f"Number of songs to select for each mid point: {n_to_selects}")

    for i in range(k):
        emo_sim_vector = emotional_sims[i]
        sorted_indexes = np.argsort(emo_sim_vector)[::-1]
        n_to_select = n_to_selects[i]
        dynamic_m = m
        if dynamic_m <= n_to_select:
            print(f"Had to increase m from {dynamic_m} to {n_to_select*2} to select enough songs.")
            dynamic_m = n_to_select*2

        available = get_available_indexes(sorted_indexes, used_indexes, dynamic_m)

        # Reordenamiento según contexto 
        ctx_sim_vector = cosine_similarity([contextual_midpoints[i]], contextual_data_embeddings[available])[0] #poner contextual_data_emb
        sorted_ctx_indexes = np.argsort(ctx_sim_vector)[::-1]
        sorted_available = [available[j] for j in sorted_ctx_indexes]
    

        # Aplicar selección final según criterio
        if selection == 'best':
            selected = sorted_available[:n_to_select]
        elif selection == 'random':
            if len(sorted_available) < n_to_select:
                raise ValueError(f"Not enough candidates to sample: {len(sorted_available)} available, {n_to_select} needed.")
            selected = random.sample(sorted_available, n_to_select)
        else:
            raise ValueError("Invalid selection mode. Use 'best' or 'random'.")

        chosen_ids.extend(selected)
        used_indexes.update(selected)

    return chosen_ids

def get_playlist_ids2_weighted(emotional_emb1,emotional_emb2,contextual_emb1, contextual_emb2, genres=[], k=2, weight_emotion=0.7, weight_context=0.3,selection='best', n=1):
    # 1) Convertir a numpy si hace falta
    for arr in (emotional_emb1, emotional_emb2, contextual_emb1, contextual_emb2):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
    # 2) Calcular mid-points
    emo_k = get_k_mid_points(emotional_emb1, emotional_emb2, k)
    ctx_k = get_k_mid_points(contextual_emb1, contextual_emb2, k)
    # 3) Similitudes por separado
    emo_sims = cosine_similarity(emo_k, emotional_data_embeddings)
    ctx_sims = cosine_similarity(ctx_k,   contextual_data_embeddings)
    # 4) Selección ponderada
    chosen_idxs = choose_ids_weighted(emo_sims, ctx_sims, k,
                                      weight_emotion, weight_context,
                                      selection, n)
    print(f"Amount of songs selected: {len(chosen_idxs)}")

    return [data_ids[i] for i in chosen_idxs]

def choose_ids_weighted(emo_sims, ctx_sims, k, w_emo, w_ctx, selection='best', n=1):
    """
    Para cada uno de los k mid-points, calcula la similitud combinada
    y selecciona las n canciones con mayor score (o muestra aleatoria
    dentro de las top-n si selection='random').
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
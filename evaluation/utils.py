import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

import os
import sys

sys.path.append(os.path.abspath(os.path.join("../..", "emotions")))
from emotions import detect_user_emotions, emotions_labels, data_embeddings, choose_ids, df, data_ids

def predict_one(text: str) -> str:
    """Devuelve la emoción top-1 de roBERTa."""
    return detect_user_emotions(text, n=1)[1][0]

def get_emotional_embedding(text: str) -> np.ndarray:
    """Devuelve la representación emocional de un texto."""
    return detect_user_emotions(text, n=1)[0]

def load_similarity_matrix(path: str = "emotion_similarity.npy") -> np.ndarray:
    """Carga (o construye) la matriz de similitud entre emociones."""
    if os.path.exists(path):
        return np.load(path)
    tokenizer   = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    model = AutoModel.from_pretrained("SamLowe/roberta-base-go_emotions", add_pooling_layer=False).eval()

    @torch.no_grad()
    def embed(sentence):
        out = model(**tokenizer(sentence, return_tensors="pt"))
        cls = out.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls, dim=-1).squeeze()

    emb = torch.stack([embed(lbl) for lbl in emotions_labels])
    S   = torch.clamp(emb @ emb.T, 0, 1).cpu().numpy()
    np.save(path, S)
    return S

def plot_confusion(cm: np.ndarray, labels, title: str, fname: str, vmax=1.0, save=True):
    """Dibuja y guarda una matriz de confusión."""
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues", vmax=vmax,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    if save:
        plt.savefig(fname, dpi=300)
    plt.show()
    plt.close()


def get_lyrics(embedding_1):
                 
    """
    """
    if not isinstance(embedding_1, np.ndarray):
        embedding_1 = np.array(embedding_1)
    if embedding_1.ndim == 1:
        embedding_1 = embedding_1.reshape(1, -1)

    similarities = cosine_similarity(embedding_1, data_embeddings)[0]
    #get the most similar song without choose_ids, only the best match
    best_idx = np.argmax(similarities)
    #its only one get_song
    closest_id = data_ids[best_idx]
    lyrics = df.loc[df["id"] == closest_id, "lyrics"].iat[0]
    return lyrics



    # closest_idxs = choose_ids(similarities, k, selection, m, mode, n)
    # closest_id = [data_ids[idx] for idx in closest_idxs]
    # song = df[df["id"].isin(closest_id)]["lyrics"].values.tolist()
    # return song
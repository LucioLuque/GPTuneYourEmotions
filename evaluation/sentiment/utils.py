import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

import os
import sys

sys.path.append(os.path.abspath(os.path.join("../..", "emotions")))
from emotions import detect_user_emotions, emotions_labels

def predict_one(text: str) -> str:
    """Devuelve la emoción top-1 de roBERTa."""
    return detect_user_emotions(text, n=1)[1][0]

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

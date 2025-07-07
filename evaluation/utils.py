import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

import os
import sys

sys.path.append(os.path.abspath(os.path.join("../..", "emotions")))
from emotions import detect_user_emotions, emotions_labels, emotional_data_embeddings, dataset, data_ids
from context.context import contextual_data_embeddings

def predict_one(text: str) -> str:
    """
    Predicts the top-1 emotion for a given text using the roBERTa model.
    --------
    Args:
        text (str): The input text for emotion detection.
    --------
    Returns:
        str: The top-1 predicted emotion label.
    """
    return detect_user_emotions(text, n=1)[1][0]

def predict_two(text: str) -> str:
    """
    Predicts the top-2 emotions for a given text using the roBERTa model.
    --------
    Args:
        text (str): The input text for emotion detection.
    --------
    Returns:
        str: The top-2 predicted emotions label.
    """
    return detect_user_emotions(text, n=2)[1]

def get_emotional_embedding(text: str) -> np.ndarray:
    """
    Generates the emotional embedding for a given text.
    --------
    Args:
        text (str): Input text to analyze.
    --------
    Returns:
        np.ndarray: Emotional embedding vector.
    """
    return detect_user_emotions(text, n=1)[0]

def load_similarity_matrix(path: str = "emotion_similarity.npy") -> np.ndarray:
    """
    Loads or constructs the similarity matrix between emotions.
    --------
    Args:
        path (str): Path to the similarity matrix file (default: "emotion_similarity.npy").
    --------
    Returns:
        np.ndarray: A 2D numpy array representing the similarity matrix.
    --------
    Notes:
        If the file exists, it loads the matrix from the file.
        Otherwise, it constructs the matrix using embeddings from the roBERTa model.
    """
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

def plot_confusion(cm: np.ndarray, labels: List[str], title: str, fname: str, vmax: float = 1.0, save: bool = True) -> None:
    """
    Plots and saves a confusion matrix as a heatmap.
    --------
    Args:
        cm (np.ndarray): The confusion matrix to plot.
        labels (list): List of labels for the axes.
        title (str): Title of the plot.
        fname (str): File path to save the plot.
        vmax (float): Maximum value for the color scale (default: 1.0).
    --------
    Returns:
        None
    """
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=False, fmt=".2f", cmap="Greens", vmax=vmax,
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

def align_predictions_with_labels(pred: list, real: list) -> Tuple[list, list]:
    """
    Aligns predicted emotions with real emotions such that matching emotions are at the same index.
    If both predictions exactly match the real labels, the original order is preserved.
    --------
    Args:
        pred (list): List of predicted emotions.
        real (list): List of real emotions.
    --------
    Returns:
        tuple: Two lists, aligned predictions and aligned real emotions.
    --------
    Notes:
        - Matching emotions are placed at the same index.
        - Remaining emotions are assigned to the next available indices.
    """
    # If both predictions exactly match the real labels, keep the order
    if set(pred) == set(real):
        return sorted(pred), sorted(real)
 
    aligned_pred = [None, None]
    aligned_real = [None, None]

    # Align matching emotions
    for i, emotion in enumerate(real):
        if emotion in pred:
            aligned_pred[i] = emotion
            aligned_real[i] = emotion

    # Find remaining predictions and real emotions
    remaining_pred = [p for p in pred if p not in aligned_pred]
    remaining_real = [r for r in real if r not in aligned_real]

    for i in range(2):
        if aligned_pred[i] is None:
            aligned_pred[i] = remaining_pred.pop(0)
        if aligned_real[i] is None:
            aligned_real[i] = remaining_real.pop(0)

    return aligned_pred, aligned_real

def get_lyrics(emotional_embedding: np.ndarray, contextual_embedding: np.ndarray, weight_emotion = 0.5, weight_context = 0.5) -> str:
    """
    Retrieves the lyrics of the song that best matches the given emotional and contextual embeddings.
    --------
    Args:
        emotional_embedding (np.ndarray): Emotional embedding vector.
        contextual_embedding (np.ndarray): Contextual embedding vector.
        weight_emotion (float): Weight for emotional similarity (default: 0.5).
        weight_context (float): Weight for contextual similarity (default: 0.5).
    --------
    Returns:
        str: Lyrics of the best matching song.
    --------
    Notes:
        - Combines emotional and contextual similarities using weighted cosine similarity.
        - Selects the song with the highest combined similarity score.
    """
    if not isinstance(emotional_embedding, np.ndarray):
        emotional_embedding = np.array(emotional_embedding)
    if emotional_embedding.ndim == 1:
        emotional_embedding = emotional_embedding.reshape(1, -1)

    if not isinstance(contextual_embedding, np.ndarray):
        contextual_embedding = np.array(contextual_embedding)
    if contextual_embedding.ndim == 1:
        contextual_embedding = contextual_embedding.reshape(1, -1)

    emotion_similarities = cosine_similarity(emotional_embedding, emotional_data_embeddings)[0]
    contextual_similarities = cosine_similarity(contextual_embedding, contextual_data_embeddings)[0]

    combined_similarities = (weight_emotion * emotion_similarities) + (weight_context * contextual_similarities)

    best_idx = np.argmax(combined_similarities)
    best_id = data_ids[best_idx]
    lyrics = dataset.loc[dataset["id"] == best_id, "lyrics"].iat[0]
    return lyrics
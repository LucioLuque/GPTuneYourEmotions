from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import no_grad
import numpy as np
import pandas as pd
import os
from typing import List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# If the file all_emotional_embeddings.npy exists in data, use it, otherwise use first_1000_emotional_embeddings.npy
if os.path.exists(os.path.join(ROOT_DIR, "data", "all_emotional_embeddings.npy")):
    EMOTIONAL_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "all_emotional_embeddings.npy")
else:
    EMOTIONAL_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "first_1000_emotional_embeddings.npy")
    
# If the file dataset.csv exists in data, use it, otherwise use first_1000_dataset.csv
if os.path.exists(os.path.join(ROOT_DIR, "data", "dataset.csv")):
    DATASET_PATH = os.path.join(ROOT_DIR, "data", "dataset.csv")
else:
    DATASET_PATH = os.path.join(ROOT_DIR, "data", "first_1000_dataset.csv")

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model     = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
model.eval()

id2label = model.config.id2label

neutral_idx = next(i for i,lbl in id2label.items() if lbl == "neutral")
_mask = np.arange(len(id2label)) != neutral_idx

emotional_data_embeddings = np.load(EMOTIONAL_EMBEDDINGS_PATH)
dataset = pd.read_csv(DATASET_PATH)
data_ids = dataset["id"]
lyrics_by_id = dict(zip(dataset["id"], dataset["lyrics"]))

filtered_labels = {
    new_i: id2label[old_i]
    for new_i, old_i in enumerate(np.where(_mask)[0])
}

emotions_labels = list(filtered_labels.values())

def get_token_chunks(text: str, tokenizer: AutoTokenizer, max_length: int = 512, stride: int = 384) -> List[np.ndarray]:
    """
    Splits the input text into token chunks of a specified maximum length with a given stride.
    --------
    Args:
        text (str): The input text to be chunked.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text.
        max_length (int): The maximum length of each chunk. Default is 512.
        stride (int): The stride to use when creating chunks. Default is 384.
    --------
    Returns:
        List[np.ndarray]: A list of token chunks.
    """
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

def format_top_emotions(probs: np.ndarray, labels: List[str], top_n: int = 1) -> List[str]:
    """
    Formats the top_k emotions from the probabilities and labels.
    --------
    Args:
        probs (np.ndarray): The probabilities of each emotion.
        labels (List[str]): The list of emotion labels.
        top_n (int): The number of top emotions to return. Default is 1.
    --------
    Returns:
        List[str]: A list of the top_n emotion labels.
    """
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]
    return [labels[i] for i in top_indices]

def get_emotional_embedding_v1(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> np.ndarray:
    """
    Computes the emotional embedding of a given text by extracting averaging the
    logits of the model for each chunk of the text.
    --------
    Args:
        text (str): The input text to be embedded.
        model (AutoModelForSequenceClassification): The pre-trained model for emotion classification.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text.
    --------
    Returns:
        np.ndarray: The average logits of the model for the input text.
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

def take_out_neutral_emotion(emotional_embedding: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Takes out the neutral emotion from the emotional embedding.
    --------
    Args:
        emotional_embedding (np.ndarray): The emotional embedding from which to remove the neutral emotion.
    --------
    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing the emotional embedding without the neutral emotion and the filtered labels.
    """
    return emotional_embedding[_mask], filtered_labels

def detect_user_emotions(user_input: str, n: int = 1) -> Tuple[np.ndarray, List[str]]:
    """
    Detects the user's emotions from the input text and returns the emotional embedding and the top emotions.
    --------
    Args:
        user_input (str): The user's input text from which to detect emotions.
        n (int): The number of top emotions to return. Default is 1.
    --------
    Returns:
        Tuple[np.ndarray, List[str]]: The emotional embedding without the neutral emotion and the top emotions.
    """
    emotional_embedding_1 = get_emotional_embedding_v1(user_input, model, tokenizer)
    emotional_embedding_2, labels_2 = take_out_neutral_emotion(emotional_embedding_1)
    top_emotion = format_top_emotions(emotional_embedding_2, labels_2, top_n=n)
    return emotional_embedding_2, top_emotion
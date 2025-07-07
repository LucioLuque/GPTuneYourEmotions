import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from typing import List
import os

context_model = SentenceTransformer("intfloat/multilingual-e5-base")
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# If the file all_context_embeddings.npy exists in data, use it, otherwise use first_1000_context_embeddings.npy
if os.path.exists(os.path.join(ROOT_DIR, "data", "all_context_embeddings.npy")):
    CONTEXTUAL_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "all_context_embeddings.npy")
else:
    CONTEXTUAL_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "data", "first_1000_context_embeddings.npy")
contextual_data_embeddings = np.load(CONTEXTUAL_EMBEDDINGS_PATH)

def chunk_text(text: str, max_tokens: int = 512, stride: int = 384) -> List[str]:
    """
    Splits the input text into chunks of a specified maximum token length with a given stride.
    --------
    Args:
        text (str): The input text to be chunked.
        max_tokens (int): The maximum number of tokens per chunk. Default is 512.
        stride (int): The number of tokens to shift for the next chunk. Default is 384.
    --------
    Returns:
        List[str]: A list of text chunks.
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    for start in range(0, len(tokens), stride):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
    return chunks

def get_context_embedding(text: str) -> np.ndarray:
    """
    Generates an average context embedding for the input text using the multilingual E5 model.
    --------
    Args:
        text (str): The input text to be embedded.
    --------
    Returns:
        np.ndarray: The average context embedding vector for the input text.
    """
    query = "query: " + text
    chunks = chunk_text(query, max_tokens=512, stride=384)
    if not chunks:
        return np.zeros(context_model.get_sentence_embedding_dimension())

    embeddings = context_model.encode(
        chunks,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=False
    )
    avg_embedding = embeddings.mean(dim=0).cpu().numpy()
    return avg_embedding
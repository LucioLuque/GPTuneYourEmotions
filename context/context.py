import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

context_model = SentenceTransformer("intfloat/multilingual-e5-base")
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

def chunk_text(text, max_tokens=512, stride=384):
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

def get_context_embedding(text):
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
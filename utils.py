import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


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

def detect_emotions(tokens, model):
    """
    Detects emotions from a list of tokens.
    Expects a tokenized input of size<=512
    Returns the probabilities
    """

    with torch.no_grad():
        outputs = model(tokens.unsqueeze(0))
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
    
    return probs

def format_top_emotions(probs, labels, top_k=5):
    """
    Formats the top_k emotions from the probabilities
    """
    emo_probs_dict = [(labels[i], probs[i]) for i in range(len(labels))]
    emo_probs_dict.sort(key=lambda x: x[1], reverse=True)
    top_emotions = emo_probs_dict[:top_k]
    
    return top_emotions

def get_top_emotions(text, model, tokenizer, top_k=5):
    token_chunks = get_token_chunks(text, tokenizer)
    
    chunk_probs = []
    labels = model.config.id2label
    for chunk in token_chunks:
        emo_probs = detect_emotions(chunk, model)
        chunk_probs.append(emo_probs)
        top_chunk_emotions = format_top_emotions(emo_probs, labels, top_k)
        print(f"Top {top_k} emotions for chunk: {top_chunk_emotions}")

    # average pooling
    avg_probs = np.mean(chunk_probs, axis=0)
    avg_probs = avg_probs.tolist()
    avg_probs = [float(x) for x in avg_probs]
    top_text_emotions = format_top_emotions(avg_probs, labels, top_k)
    print(f"Top {top_k} emotions for text: {top_text_emotions}")

def get_emotional_embedding_v1(text, model, tokenizer):
    """ 
    Computes the emotional embedding of a given text by extracting averagint the 
    logits of the model for each chunk of the text.
    """
    token_chunks = get_token_chunks(text, tokenizer)
    
    chunk_logits = []
    for chunk in token_chunks:
        with torch.no_grad():
            outputs = model(chunk.unsqueeze(0))
            logits = outputs.logits
            chunk_logits.append(logits.squeeze().numpy())
    
    avg_logits = np.mean(chunk_logits, axis=0)
    return avg_logits

def get_emotional_embedding_v2(text, model, tokenizer):
    """Esta versión es del chat, supuestamente es mucho más rápida."""
    token_chunks = get_token_chunks(text, tokenizer)

    # Convert list of token tensors into padded batch
    input_batch = torch.nn.utils.rnn.pad_sequence(token_chunks, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_masks = (input_batch != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model(input_batch, attention_mask=attention_masks)
        logits = outputs.logits

    # Promediar logits por chunk
    avg_logits = logits.mean(dim=0).numpy()
    return avg_logits

def get_emotional_embeddings_v3(texts, model, tokenizer, device='cpu'):
    """
    Procesa múltiples textos a la vez (batching real de samples).
    Devuelve un array de embeddings emocionales, uno por texto.
    """
    model.to(device)
    model.eval()

    all_embeddings = []

    for text in tqdm(texts, desc="Procesando textos"):
        token_chunks = get_token_chunks(text, tokenizer)

        # Convertir chunks a batch con padding
        input_batch = torch.nn.utils.rnn.pad_sequence(
            token_chunks, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)

        attention_masks = (input_batch != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model(input_batch, attention_mask=attention_masks)
            logits = outputs.logits  # shape: (num_chunks, num_classes)

        avg_logits = logits.mean(dim=0).cpu().numpy()
        all_embeddings.append(avg_logits)

    return np.stack(all_embeddings)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import no_grad
from numpy import mean, delete, arange, where

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model     = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
model.eval()

id2label = model.config.id2label


neutral_idx = next(i for i,lbl in id2label.items() if lbl == "neutral")
_mask = arange(len(id2label)) != neutral_idx

# crea también el diccionario filtrado de etiquetas (índices renumerados)
filtered_labels = {
    new_i: id2label[old_i]
    for new_i, old_i in enumerate(where(_mask)[0])
}

emotions_labels = list(filtered_labels.values())

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
    
    avg_logits = mean(chunk_logits, axis=0)
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
    
import os, sys, torch, numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emotions.emotions import detect_user_emotions, emotions_labels
IDX = {lbl: i for i, lbl in enumerate(emotions_labels)}
RESULTS_STORAGE_PATH = os.path.join(os.path.dirname(__file__), "sentiment_results")
BASE_DIR = os.path.dirname(__file__)

def predict_one(text: str) -> str:
    """Devuelve la emoción top-1 de roBERTa."""
    return detect_user_emotions(text, n=1)[1][0]

def load_similarity_matrix(path: str = "emotion_similarity.npy") -> np.ndarray:
    """Carga (o construye) la matriz de similitud entre emociones."""
    if os.path.exists(path):
        return np.load(path)

    tok   = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    model = AutoModel.from_pretrained(
        "SamLowe/roberta-base-go_emotions", output_hidden_states=True
    ).eval()

    @torch.no_grad()
    def embed(sentence):
        out = model(**tok(sentence, return_tensors="pt"))
        cls = out.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls, dim=-1).squeeze()

    emb = torch.stack([embed(lbl) for lbl in emotions_labels])
    S   = torch.clamp(emb @ emb.T, 0, 1).cpu().numpy()
    np.save(path, S)
    return S

def plot_confusion(cm: np.ndarray, labels, title: str, fname: str, vmax=1.0):
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
    plt.show()
    plt.savefig(fname, dpi=300)
    plt.close()

def main():
    
    df = pd.read_csv(os.path.join(BASE_DIR, "eval_prompts.csv"))
    df["pred"] = df["text"].apply(predict_one)

    y_true = df["label"].values
    y_pred = df["pred"].values

    hard_acc = accuracy_score(y_true, y_pred)
    hard_cm  = confusion_matrix(y_true, y_pred, labels=emotions_labels)

    print(f"Hard accuracy : {hard_acc:.3f}")
    plot_confusion(hard_cm, emotions_labels,
                   title="Hard Confusion Matrix",
                   fname=os.path.join(RESULTS_STORAGE_PATH, "cm_hard.png"), vmax=hard_cm.max())

    S = load_similarity_matrix()
    soft_scores = [S[IDX[t], IDX[p]] for t, p in zip(y_true, y_pred)]
    soft_acc    = np.mean(soft_scores)
    print(f"Soft accuracy : {soft_acc:.3f}")

    soft_cm = np.zeros_like(S)
    for t, p in zip(y_true, y_pred):
        soft_cm[IDX[t], IDX[p]] += S[IDX[t], IDX[p]]

    plot_confusion(soft_cm, emotions_labels,
                   title="Soft Confusion Matrix",
                   fname=os.path.join(RESULTS_STORAGE_PATH, "cm_soft.png"), vmax=1.0)

    df.to_csv(os.path.join(RESULTS_STORAGE_PATH, "eval_prompts_with_preds.csv"), index=False)

if __name__ == "__main__":
    main()

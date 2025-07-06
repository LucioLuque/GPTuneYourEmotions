import os, sys, torch, numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emotions.emotions import detect_user_emotions, emotions_labels
IDX = {lbl: i for i, lbl in enumerate(emotions_labels)}
RESULTS_STORAGE_PATH = os.path.join(os.path.dirname(__file__), "sentiment_results")
BASE_DIR = os.path.dirname(__file__)

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

    tok   = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    model = AutoModel.from_pretrained(
        "SamLowe/roberta-base-go_emotions", output_hidden_states=True
    ).eval()

    @torch.no_grad()
    def embed(sentence): # Generates an embedding for a given sentence using the roBERTa model
        out = model(**tok(sentence, return_tensors="pt"))
        cls = out.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls, dim=-1).squeeze()

    emb = torch.stack([embed(lbl) for lbl in emotions_labels])
    S   = torch.clamp(emb @ emb.T, 0, 1).cpu().numpy()
    np.save(path, S)
    return S

def plot_confusion(cm: np.ndarray, labels, title: str, fname: str, vmax=1.0):
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
    sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues", vmax=vmax,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
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
    
    print("=== Classification report ===")
    report = classification_report(df["label"], df["pred"], digits=3)
    report_path = os.path.join(RESULTS_STORAGE_PATH, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

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

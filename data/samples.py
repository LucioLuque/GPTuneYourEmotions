import numpy as np
import pandas as pd
import os

CURRENT_DIR = os.path.dirname(__file__)
all_emotional_embeddings = np.load(os.path.join(CURRENT_DIR, "all_emotional_embeddings.npy"))
all_context_embeddings = np.load(os.path.join(CURRENT_DIR, "all_context_embeddings.npy"))
dataset = pd.read_csv(os.path.join(CURRENT_DIR, "dataset.csv"))

# Save only the first 1000 rows
np.save("data/first_1000_emotional_embeddings.npy", all_emotional_embeddings[:1000])
np.save("data/first_1000_context_embeddings.npy", all_context_embeddings[:1000])
dataset[:1000].to_csv("data/first_1000_dataset.csv", index=False)
import code
import wordfreq
import nltk

import torch
import clip
import numpy as np


with open('tmp/caps.txt', 'r') as f:
    candidates = f.read().split('\n')[:80]

model, preprocess = clip.load("ViT-B/32", device="cpu")

best = (None, float('inf'))

B = np.load('laion1m_clip_embeddings.npy')
B /= np.linalg.norm(B, axis=0)
B = B.T

M_B = B.mean(axis=1, keepdims=True)
C = np.cov(B)
base_R = B - M_B
base_R_SQ_Norm = np.linalg.norm(base_R*base_R, axis=1)
base_RMSE = np.sqrt(np.mean(base_R_SQ_Norm))

subset = []
scores = []

for word in candidates:
    clip_toks = clip.tokenize([word]).to("cpu")

    A = model.encode_text(clip_toks).detach().numpy().T
    A /= np.linalg.norm(A, axis=0)


    S = A.T @ B

    K = (C @ A) @ np.linalg.inv(A.T @ C @ A)

    B_hat = M_B + K @ (S - A.T @ M_B)

    R = B - B_hat
    R_SQ_Norm = np.linalg.norm(R*R, axis = 1)
    RMSE = np.sqrt(np.mean(R_SQ_Norm))

    if RMSE < best[1]:
        best = (word, RMSE)
    scores.append((word, RMSE))

    print(f"Keyword: {word}, RMSE: {RMSE:.4f} ({RMSE-base_RMSE:.4f} from base RMSE)")

ranked = sorted(scores, key=lambda x: x[1])
with open('tmp/scores.txt', 'w') as f:
    for word, _score in ranked:
        f.write(f"{word}\n")



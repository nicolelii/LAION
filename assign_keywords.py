import torch
import clip
import numpy as np
from tqdm import tqdm

# Load CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define candidate keyword list
keywords = [
    "animal", "scary", "funny", "person", "object", "vehicle", "outdoor", "indoor",
    "tool", "art", "food", "technology", "plant", "building", "landscape", "water",
    "sports", "fashion", "cute", "cartoon", "city", "moon", "flower", "sunset",
    "abstract", "dark", "bright", "sad", "happy", "smile"
]

# Encode all keyword texts
with torch.no_grad():
    keyword_tokens = clip.tokenize(keywords).to(device)  # [30, 77]
    keyword_embeddings = model.encode_text(keyword_tokens)  # [30, 512]
    keyword_embeddings /= keyword_embeddings.norm(dim=-1, keepdim=True)  # Normalize

# Load image embeddings and filenames
image_embeddings = np.load("laion1m_clip_embeddings.npy")  # [N, 512]
with open("laion1m_filenames.txt", "r") as f:
    filenames = [line.strip() for line in f]

assert image_embeddings.shape[0] == len(filenames)

# Assign top-3 keywords for each image
output_lines = []
image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32, device=device)
image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)  # Normalize

batch_size = 10000
for i in tqdm(range(0, len(filenames), batch_size)):
    batch_emb = image_embeddings[i:i+batch_size]  # [B, 512]
    similarity = batch_emb @ keyword_embeddings.T  # [B, 30]

    topk = torch.topk(similarity, k=3, dim=1)  # top-3 keyword indices
    top_indices = topk.indices.cpu().numpy()  # [B, 3]

    for j, indices in enumerate(top_indices):
        fname = filenames[i + j]
        tags = [keywords[k] for k in indices]
        output_lines.append(f"{fname}: {', '.join(tags)}")

# Save
with open("laion1m_keywords.txt", "w") as f:
    for line in output_lines:
        f.write(f"{line}\n")

print("Finished assigning keywords to all images.")

import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

TARGET_IMAGE_COUNT = 1000448
BATCH_SIZE = 1000
NUM_WORKERS = 16

class ImageDataset(Dataset):
    def __init__(self, image_folder, preprocess, max_images):
        # Recursively find all .jpg images
        all_images = []
        for dp, _, filenames in os.walk(image_folder):
            for f in filenames:
                if f.lower().endswith(".jpg"):
                    all_images.append(os.path.join(dp, f))
                    if len(all_images) >= max_images:
                        break
            if len(all_images) >= max_images:
                break

        self.image_files = sorted(all_images)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        try:
            image = Image.open(path).convert("RGB")
            return self.preprocess(image), os.path.basename(path)
        except Exception:
            return None, None  # Skip corrupted

def collate_fn(batch):
    images, filenames = zip(*batch)
    images = [img for img in images if img is not None]
    filenames = [fn for fn in filenames if fn is not None]
    if len(images) == 0:
        return None, None
    return torch.stack(images), filenames

# Setup
image_folder = "/home/yl3558/laion/laion3m_data"
device = "cuda" 
model, preprocess = clip.load("ViT-B/32", device=device)

# Dataset and DataLoader
dataset = ImageDataset(image_folder, preprocess, max_images=TARGET_IMAGE_COUNT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    shuffle=False, collate_fn=collate_fn)

# Embedding loop
all_embeddings = []
all_filenames = []
with torch.no_grad():
    for images, filenames in tqdm(loader, total=(TARGET_IMAGE_COUNT + BATCH_SIZE - 1) // BATCH_SIZE):
        if images is None:
            continue
        image_input = images.to(device)
        features = model.encode_image(image_input).cpu().numpy()
        all_embeddings.append(features)
        all_filenames.extend(filenames)
        if len(all_filenames) >= TARGET_IMAGE_COUNT:
            break

# Save outputs
final_embeddings = np.vstack(all_embeddings)[:TARGET_IMAGE_COUNT]
np.save("laion1m_clip_embeddings.npy", final_embeddings)

with open("laion1m_filenames.txt", "w") as f:
    for fname in all_filenames[:TARGET_IMAGE_COUNT]:
        f.write(f"{fname}\n")

print(f"✅ Done: embedded {len(final_embeddings)} images.")

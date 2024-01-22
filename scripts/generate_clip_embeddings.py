from typing import Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

from model.clip_module import CLIPVisionTower

dataset_name: str = "HuggingFaceM4/COCO"
MODEL_ID: str = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
max_size: Tuple[int, int] = (512, 512)
batch_size: int = 256
device: str = "cuda" if torch.cuda.is_available() else "cpu"

ds = load_dataset(dataset_name)

train_ds = ds["train"]
val_ds = ds["validation"]
vision_tower = CLIPVisionTower(MODEL_ID).to(device)


def build_embeddings_data(dataset, savename):
    embeddings = {}
    imgzs = []
    filenames = []
    for p in tqdm(dataset):
        img0 = p["image"]
        img0.thumbnail(max_size)

        imgzs.append(img0)
        filenames.append(p["filename"])
        if len(imgzs) == batch_size:
            batch_embeddings = vision_tower(imgzs)
            for filename, embedding in zip(filenames, batch_embeddings):
                embeddings[filename] = embedding

            imgzs = []
            filename = []

    torch.save(embeddings, savename)


print("[x] Building train embeddings...")
build_embeddings_data(train_ds, "data/train_embeddings.pt")

print("[x] Building validation embeddings...")
build_embeddings_data(val_ds, "data/val_embeddings.pt")

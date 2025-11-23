"""
retrieve.py
Run retrieval on trained model using text queries like:
    python retrieve.py "class=bottle color=blue material=plastic size=medium"
"""

import torch
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader

from dataset import EverydayObjectsDataset
from model import MultiTaskViT

DEVICE = "cpu"
DATA_ROOT = "data"
MODEL_PATH = "outputs_pooled/best_model.pth"
ATTR_FILE = "attributes_pooled.yaml"

def parse_query(q, class_to_idx, attr_schema):
    tokens = q.lower().split()
    result = {k: None for k in ["class", "color", "material", "condition", "size"]}

    # Build a lowercase mapping for class names so queries are case-insensitive
    class_to_idx_lower = {k.lower(): v for k, v in class_to_idx.items()}

    for tok in tokens:
        if "=" in tok:
            key, value = tok.split("=")
            value = value.strip()
            if key == "class":
                # Look up using the lowercase mapping
                if value in class_to_idx_lower:
                    result["class"] = class_to_idx_lower[value]
            else:
                if key in attr_schema and value in attr_schema[key]:
                    result[key] = attr_schema[key].index(value)
    return result


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@torch.no_grad()
def build_index(model, loader):
    index = []
    model.eval()

    for batch in loader:
        imgs, _, _, _, _, _, rows = batch
        imgs = imgs.to(DEVICE)

        feats, class_logits, attr_logits = model(imgs)

        preds_class = class_logits.argmax(dim=1).cpu().numpy()
        preds_attr = {
            name: attr_logits[name].argmax(dim=1).cpu().numpy()
            for name in ["color", "material", "condition", "size"]
        }

        # `rows` is a dict of lists (one list per metadata field) from the DataLoader collate
        num_items = len(next(iter(rows.values())))
        for i in range(num_items):
            meta = {k: rows[k][i] for k in rows}

            index.append({
                "feat": feats[i].cpu().numpy(),
                "pred_class": int(preds_class[i]),
                "pred_color": int(preds_attr["color"][i]),
                "pred_material": int(preds_attr["material"][i]),
                "pred_condition": int(preds_attr["condition"][i]),
                "pred_size": int(preds_attr["size"][i]),
                "meta": meta
            })

    return index


def retrieve(index, query_dict, top_k=5):
    scores = []

    for item in index:
        score = 0

        if query_dict["class"] is not None and item["pred_class"] == query_dict["class"]:
            score += 1.0

        for attr in ["color", "material", "condition", "size"]:
            qv = query_dict[attr]
            if qv is not None and item[f"pred_{attr}"] == qv:
                score += 0.5

        scores.append(score)

    ranked = np.argsort(scores)[::-1][:top_k]
    return [index[i] for i in ranked if scores[i] > 0]


def main():
    # Load dataset
    full_ds = EverydayObjectsDataset(DATA_ROOT, split="val")
    loader = DataLoader(full_ds, batch_size=32, shuffle=False)

    # Load attributes
    with open(os.path.join(DATA_ROOT, ATTR_FILE)) as f:
        attr_schema = yaml.safe_load(f)

    model = MultiTaskViT(
        "deit_tiny_patch16_224",
        num_classes=len(full_ds.class_to_idx),
        attr_schema=attr_schema,
        pretrained=False,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Loaded fine-tuned model!")

    # Build index
    print("Indexing images...")
    index = build_index(model, loader)

    # Run sample query
    q = input("\nEnter query (e.g., 'class=bottle color=blue material=plastic'):\n> ")
    q_dict = parse_query(q, full_ds.class_to_idx, attr_schema)
    print(full_ds.class_to_idx)
    print("\nRetrieving...")
    results = retrieve(index, q_dict, top_k=5)

    from PIL import Image

    print("\nTop Results:")
    if not results:
        print("No matching results.")
        return

    # Load images
    images = []
    for r in results:
        img_rel = r['meta']['image_path']
        img_path = os.path.join(DATA_ROOT, img_rel)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))  # make all thumbnails same size
            images.append(img)
        except FileNotFoundError:
            print(f"Missing image: {img_path}")

    if not images:
        print("No images could be loaded.")
        return

    # Arrange in a grid (up to 3 columns)
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols

    thumb_w, thumb_h = images[0].size
    grid_w = cols * thumb_w
    grid_h = rows * thumb_h

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * thumb_w
        y = row * thumb_h
        grid.paste(img, (x, y))

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "retrieved_grid.jpg")
    grid.save(out_path)
    grid.show()
    print(f"Displayed grid: {out_path}")


if __name__ == "__main__":
    main()

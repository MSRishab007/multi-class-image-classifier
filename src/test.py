"""
test.py

Loads the fine-tuned DeiT-Tiny model (outputs/best_model.pth)
and evaluates it on a chosen split ('test' or 'val').

Also shows how to print some example predictions.
"""

import os
import yaml
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset import EverydayObjectsDataset
from model import MultiTaskViT

# ---- CONFIG ----
DATA_ROOT = "data"                  # same as in train.py
ATTR_YAML_NAME = "attributes.yaml"
LABELS_CSV_NAME = "labels.csv"
MODEL_PATH = "outputs_vit_tiny/best_model.pth"
BACKBONE_NAME = "deit_tiny_patch16_224"   # this is your DeiT-Tiny
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cpu")        # CPU-only


ATTR_NAMES = ["color", "material", "condition", "size"]


@torch.no_grad()
def evaluate(model, loader, idx_to_class):
    model.eval()

    all_true = []
    all_pred = []

    all_true_attr = {n: [] for n in ATTR_NAMES}
    all_pred_attr = {n: [] for n in ATTR_NAMES}

    for batch in loader:
        imgs, y_class, y_color, y_mat, y_cond, y_size, rows = batch
        imgs = imgs.to(DEVICE)

        feats, class_logits, attr_logits = model(imgs)

        preds_class = class_logits.argmax(dim=1).cpu().numpy()
        all_true.extend(y_class.numpy())
        all_pred.extend(preds_class)

        ys = {
            "color": y_color,
            "material": y_mat,
            "condition": y_cond,
            "size": y_size,
        }
        for name in ATTR_NAMES:
            true_attr = ys[name].numpy()
            pred_attr = attr_logits[name].argmax(dim=1).cpu().numpy()
            all_true_attr[name].extend(true_attr)
            all_pred_attr[name].extend(pred_attr)

    acc = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, average="macro")
    f1_per_class = f1_score(all_true, all_pred, average=None)

    print(f"\n=== TEST RESULTS ===")
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Macro F1:         {f1_macro:.4f}")
    print("\nPer-class F1:")
    for i, f1 in enumerate(f1_per_class):
        print(f"  {idx_to_class[i]}: {f1:.4f}")

    print("\nAttribute accuracies:")
    for name in ATTR_NAMES:
        from sklearn.metrics import accuracy_score as attr_acc
        acc_attr = attr_acc(all_true_attr[name], all_pred_attr[name])
        print(f"  {name}: {acc_attr:.4f}")

    cm = confusion_matrix(all_true, all_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)


def load_model_and_data(split="test"):
    # Dataset
    ds = EverydayObjectsDataset(
        root=DATA_ROOT,
        split=split,
        attrs_yaml=ATTR_YAML_NAME,
        labels_csv=LABELS_CSV_NAME,
        img_size=IMG_SIZE,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # Attribute schema
    with open(os.path.join(DATA_ROOT, ATTR_YAML_NAME)) as f:
        attr_schema = yaml.safe_load(f)

    num_classes = len(ds.class_to_idx)

    # Model
    model = MultiTaskViT(
        backbone_name=BACKBONE_NAME,
        num_classes=num_classes,
        attr_schema=attr_schema,
        pretrained=False,          # important! we're loading our own weights
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded fine-tuned weights from {MODEL_PATH}")

    return model, ds, loader, attr_schema


def test_split(split="test"):
    """
    Call this to test on a given split.
    Use split='test' if you have that in labels.csv,
    otherwise use split='val'.
    """
    print(f"\nTesting on split = '{split}'")
    model, ds, loader, attr_schema = load_model_and_data(split=split)
    evaluate(model, loader, ds.idx_to_class)


# ------------- single-image demo -------------

from PIL import Image
from torchvision import transforms

def predict_single_image(image_path):
    """
    Run prediction (class + attributes) on a single image file.
    """
    print(f"\nPredicting on image: {image_path}")

    # we need dataset just to reuse mappings
    ds = EverydayObjectsDataset(
        root=DATA_ROOT,
        split="val",      # any split is fine, just to load mappings
        attrs_yaml=ATTR_YAML_NAME,
        labels_csv=LABELS_CSV_NAME,
        img_size=IMG_SIZE,
    )

    with open(os.path.join(DATA_ROOT, ATTR_YAML_NAME)) as f:
        attr_schema = yaml.safe_load(f)

    num_classes = len(ds.class_to_idx)
    print((ds.class_to_idx))
    model = MultiTaskViT(
        backbone_name=BACKBONE_NAME,
        num_classes=num_classes,
        attr_schema=attr_schema,
        pretrained=False,
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats, class_logits, attr_logits = model(x)

    class_idx = class_logits.argmax(dim=1).item()
    class_name = ds.idx_to_class[class_idx]

    attr_preds = {}
    for name in ATTR_NAMES:
        idx = attr_logits[name].argmax(dim=1).item()
        attr_preds[name] = attr_schema[name][idx]

    print(f"Predicted class: {class_name}")
    print("Predicted attributes:")
    for k, v in attr_preds.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    # 1) Test on a split
    #    If you don't have 'test' in labels.csv, change to 'val'
    # test_split(split="val")   # or "val"

    predict_single_image(r"C:\Users\Hp\Downloads\DLP\DLP\data\images\team9_cap_07_b.jpg")

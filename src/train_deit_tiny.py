"""
train.py

End-to-end training script for:
- 10-class object classification
- Attribute prediction (color, material, condition, size)
- Saving best model
- Computing accuracy, F1, confusion matrix
- t-SNE visualization of feature space

Before running:
    pip install torch torchvision timm scikit-learn matplotlib pyyaml
"""

import os
import yaml
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE

from dataset import EverydayObjectsDataset
from model import MultiTaskViT

# -----------------------
# CONFIG
# -----------------------
DATA_ROOT = "data"  # folder containing images/, labels.csv, attributes.yaml
LABELS_CSV_NAME = "labels.csv"
ATTR_YAML_NAME = "attributes.yaml"

BACKBONE_NAME = "deit_tiny_patch16_224"     # or "mobilevit_xxs_100"
IMG_SIZE = 224                              # ViT input size (we'll resize from 96x96)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
W_ATTR = 0.25                               # weight for attribute losses
OUTPUT_DIR = "outputs"

DEVICE = torch.device("cpu")                # CPU-only, as requested

ATTR_NAMES = ["color", "material", "condition", "size"]


# -----------------------
# UTILS
# -----------------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(cm, classes, out_path, normalize=True):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def tsne_plot(features, labels, idx_to_class, out_path):
    tsne = TSNE(
    n_components=2,
    init="random",
    learning_rate="auto",
    perplexity=min(20, len(labels) - 1),
    verbose=1,
)
    emb_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    labels = np.array(labels)
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            s=10,
            alpha=0.7,
            label=idx_to_class[int(c)],
        )
    plt.legend(markerscale=2)
    plt.title("t-SNE of ViT Features (val set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------
# TRAIN / EVAL
# -----------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion_ce,
    epoch,
    device=DEVICE,
    w_attr=W_ATTR,
):
    model.train()
    running_loss = 0.0

    for batch in loader:
        (
            imgs,
            y_class,
            y_color,
            y_mat,
            y_cond,
            y_size,
            _,
        ) = batch

        imgs = imgs.to(device)
        y_class = y_class.to(device)
        y_color = y_color.to(device)
        y_mat = y_mat.to(device)
        y_cond = y_cond.to(device)
        y_size = y_size.to(device)

        optimizer.zero_grad()
        feats, class_logits, attr_logits = model(imgs)

        loss_class = criterion_ce(class_logits, y_class)
        loss_color = criterion_ce(attr_logits["color"], y_color)
        loss_mat = criterion_ce(attr_logits["material"], y_mat)
        loss_cond = criterion_ce(attr_logits["condition"], y_cond)
        loss_size = criterion_ce(attr_logits["size"], y_size)

        loss = loss_class + w_attr * (
            loss_color + loss_mat + loss_cond + loss_size
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    model.eval()

    all_true_class = []
    all_pred_class = []

    all_true_attr = {n: [] for n in ATTR_NAMES}
    all_pred_attr = {n: [] for n in ATTR_NAMES}

    misclassified = []

    for batch in loader:
        (
            imgs,
            y_class,
            y_color,
            y_mat,
            y_cond,
            y_size,
            rows,
        ) = batch

        imgs = imgs.to(device)
        feats, class_logits, attr_logits = model(imgs)

        # class predictions
        preds_class = class_logits.argmax(dim=1).cpu()

        # store class metrics
        all_true_class.extend(y_class.numpy())
        all_pred_class.extend(preds_class.numpy())

        # attribute predictions
        y_dict = {
            "color": y_color,
            "material": y_mat,
            "condition": y_cond,
            "size": y_size,
        }
        for name in ATTR_NAMES:
            true_attr = y_dict[name].numpy()
            pred_attr = attr_logits[name].argmax(dim=1).cpu().numpy()
            all_true_attr[name].extend(true_attr)
            all_pred_attr[name].extend(pred_attr)

        # track misclassified examples
        for i in range(len(rows)):
            if preds_class[i].item() != y_class[i].item():
                # rows is a tuple of dicts when batched by default collate_fn
                # Each element is a dict, but may need to handle nested structure
                try:
                    if isinstance(rows, (list, tuple)):
                        row_dict = rows[i] if i < len(rows) else {}
                    else:
                        row_dict = rows
                    img_path = row_dict.get("image_path", "unknown") if isinstance(row_dict, dict) else "unknown"
                except (IndexError, KeyError, TypeError):
                    img_path = "unknown"
                
                misclassified.append(
                    {
                        "image_path": img_path,
                        "true_idx": int(y_class[i].item()),
                        "pred_idx": int(preds_class[i].item()),
                    }
                )

    # --- metrics ---
    overall_acc = accuracy_score(all_true_class, all_pred_class)
    overall_f1_macro = f1_score(
        all_true_class, all_pred_class, average="macro"
    )
    f1_per_class = f1_score(all_true_class, all_pred_class, average=None)

    attr_acc = {}
    for name in ATTR_NAMES:
        acc_attr = accuracy_score(all_true_attr[name], all_pred_attr[name])
        attr_acc[name] = acc_attr

    return {
        "overall_acc": overall_acc,
        "overall_f1_macro": overall_f1_macro,
        "f1_per_class": f1_per_class.tolist(),
        "y_true": all_true_class,
        "y_pred": all_pred_class,
        "attr_acc": attr_acc,
        "misclassified": misclassified,
    }


@torch.no_grad()
def extract_val_features(model, loader, device=DEVICE):
    model.eval()
    all_feats = []
    all_labels = []

    for batch in loader:
        imgs, y_class, *_ = batch
        imgs = imgs.to(device)
        feats, _, _ = model(imgs)
        all_feats.append(feats.cpu().numpy())
        all_labels.extend(y_class.numpy())

    return np.concatenate(all_feats, axis=0), np.array(all_labels)


# -----------------------
# MAIN
# -----------------------

def main():
    ensure_dir(OUTPUT_DIR)

    # ---- DATASETS & DATALOADERS ----
    print("Loading datasets...")

    train_ds = EverydayObjectsDataset(
        DATA_ROOT,
        split="train",
        attrs_yaml=ATTR_YAML_NAME,
        labels_csv=LABELS_CSV_NAME,
        img_size=IMG_SIZE,
    )
    val_ds = EverydayObjectsDataset(
        DATA_ROOT,
        split="val",
        attrs_yaml=ATTR_YAML_NAME,
        labels_csv=LABELS_CSV_NAME,
        img_size=IMG_SIZE,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # ---- ATTRIBUTE SCHEMA ----
    with open(os.path.join(DATA_ROOT, ATTR_YAML_NAME)) as f:
        attr_schema = yaml.safe_load(f)

    # ---- MODEL ----
    num_classes = len(train_ds.class_to_idx)
    print(f"Num classes: {num_classes}")
    print(f"Backbone: {BACKBONE_NAME}")

    model = MultiTaskViT(
        backbone_name=BACKBONE_NAME,
        num_classes=num_classes,
        attr_schema=attr_schema,
        pretrained=True,
    ).to(DEVICE)

    # ---- TRAINING OBJECTS ----
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    idx_to_class = train_ds.idx_to_class

    # ---- TRAINING LOOP ----
    if os.path.exists(best_model_path):
        print(f"Found existing best model at {best_model_path}. Skipping training and going straight to evaluation.")
    else:
        print("Starting training...")
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion_ce,
                epoch=epoch,
                device=DEVICE,
                w_attr=W_ATTR,
            )
            print(f"  Train loss: {train_loss:.4f}")

            val_results = evaluate(model, val_loader, device=DEVICE)

            print(f"  Val accuracy:  {val_results['overall_acc']:.4f}")
            print(f"  Val macro-F1:  {val_results['overall_f1_macro']:.4f}")
            print("  Attribute accuracies:")
            for name, acc in val_results["attr_acc"].items():
                print(f"    {name}: {acc:.4f}")

            # save best model
            if val_results["overall_acc"] > best_acc:
                best_acc = val_results["overall_acc"]
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> New best model saved to {best_model_path}")

            scheduler.step()

        print("\nTraining complete.")
        print(f"Best val accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()

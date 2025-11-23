
import os
import sys
import base64
from functools import lru_cache

from flask import Flask, request, redirect, url_for, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import yaml
import numpy as np

# ---------------------------------
# PATHS: make src/ importable
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../DLP/web_app
PROJECT_ROOT = os.path.dirname(BASE_DIR)                       # .../DLP
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)                                       # so we can import dataset, model

from dataset import EverydayObjectsDataset
from model import MultiTaskViT

# ---------------------------------
# GLOBAL CONFIG
# ---------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLASSES_FILE = os.path.join(DATA_DIR, "classes.txt")  # 10 mapped classes
DEVICE = torch.device("cpu")
IMG_SIZE = 224

# Map (model_name, dataset_variant) -> config
# model_name: "deit" | "vit"
# dataset_variant: "ours" | "pooled"
MODEL_CONFIG = {
    # DeiT-Tiny on ORIGINAL dataset
    ("deit", "ours"): {
        "backbone": "deit_tiny_patch16_224",
        "weights_path": os.path.join(PROJECT_ROOT, "outputs", "best_model.pth"),
        "data_root": DATA_DIR,
        "labels_csv": "labels.csv",
        "attrs_yaml": "attributes.yaml",
    },
    # DeiT-Tiny on POOLED dataset
    ("deit", "pooled"): {
        "backbone": "deit_tiny_patch16_224",
        "weights_path": os.path.join(PROJECT_ROOT, "outputs_pooled", "best_model.pth"),
        "data_root": DATA_DIR,
        "labels_csv": "processed_dataset.csv",      # pooled CSV
        "attrs_yaml": "attributes_pooled.yaml",     # pooled attributes
    },
    # ViT-Tiny on ORIGINAL dataset
    ("vit", "ours"): {
        "backbone": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "weights_path": os.path.join(PROJECT_ROOT, "outputs_vit_tiny", "best_model.pth"),
        "data_root": DATA_DIR,
        "labels_csv": "labels.csv",
        "attrs_yaml": "attributes.yaml",
    },
    # ViT-Tiny on POOLED dataset
    ("vit", "pooled"): {
        "backbone": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "weights_path": os.path.join(PROJECT_ROOT, "outputs_vit_tiny_pooled", "best_model.pth"),
        "data_root": DATA_DIR,
        "labels_csv": "processed_dataset.csv",
        "attrs_yaml": "attributes_pooled.yaml",
    },
}

ATTR_NAMES = ["color", "material", "condition", "size"]

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# MODEL + INDEX LOADING
# -----------------------------

class ModelBundle:
    """
    Holds everything we need for a given (model, dataset_variant) pair:
    - model (MultiTaskViT)
    - class mappings
    - attr schema
    - full dataset + index for retrieval
    """
    def __init__(self, model, attr_schema, train_ds, val_ds, index,
                 class_to_idx, idx_to_class):
        self.model = model
        self.attr_schema = attr_schema
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.index = index
        self.class_to_idx = class_to_idx    # from classes.txt
        self.idx_to_class = idx_to_class    # from classes.txt


@lru_cache(maxsize=None)
def load_model_bundle(model_name: str, dataset_variant: str) -> ModelBundle:
    """
    Loads model + datasets + builds retrieval index for a given choice.
    Cached so we only do this once per combination while the server runs.
    """
    key = (model_name, dataset_variant)
    if key not in MODEL_CONFIG:
        raise ValueError(f"No config for {key}")

    cfg = MODEL_CONFIG[key]
    data_root = cfg["data_root"]
    labels_csv = cfg["labels_csv"]
    attrs_yaml = cfg["attrs_yaml"]

    # 1) Datasets (we only need them for images / attributes, not label mapping)
    train_ds = EverydayObjectsDataset(
        data_root,
        split="train",
        attrs_yaml=attrs_yaml,
        labels_csv=labels_csv,
        img_size=IMG_SIZE,
    )
    val_ds = EverydayObjectsDataset(
        data_root,
        split="val",
        attrs_yaml=attrs_yaml,
        labels_csv=labels_csv,
        img_size=IMG_SIZE,
    )

    # 2) Attribute schema
    with open(os.path.join(data_root, attrs_yaml)) as f:
        attr_schema = yaml.safe_load(f)

    # 3) Class names from classes.txt (10 mapped classes)
    with open(CLASSES_FILE) as f:
        class_names = [line.strip() for line in f if line.strip()]

    num_classes = len(class_names)
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    class_to_idx = {name: i for i, name in idx_to_class.items()}

    # 4) Model with correct num_classes
    backbone = cfg["backbone"]
    model = MultiTaskViT(
        backbone_name=backbone,
        num_classes=num_classes,
        attr_schema=attr_schema,
        pretrained=True,
    )
    state_dict = torch.load(cfg["weights_path"], map_location=DEVICE)
    model.load_state_dict(state_dict)   # now shapes match (10 classes)
    model.to(DEVICE)
    model.eval()

    # 5) Build retrieval index (features + predicted labels)
    full_ds = ConcatDataset([train_ds, val_ds])
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)

    index = build_index(model, full_loader)

    return ModelBundle(
        model=model,
        attr_schema=attr_schema,
        train_ds=train_ds,
        val_ds=val_ds,
        index=index,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
    )


@torch.no_grad()
def build_index(model, loader):
    """
    For retrieval: precompute features & predicted labels for all images.
    Stores image paths RELATIVE to DATA_DIR so /data/<path> works.
    """
    model.eval()
    index = []
    dataset_idx = 0  # running index over the underlying dataset

    for batch in loader:
        imgs, y_class, y_color, y_mat, y_cond, y_size, _rows = batch
        imgs = imgs.to(DEVICE)

        feats, class_logits, attr_logits = model(imgs)

        preds_class = class_logits.argmax(dim=1).cpu().numpy()
        preds_attr = {
            "color": attr_logits["color"].argmax(dim=1).cpu().numpy(),
            "material": attr_logits["material"].argmax(dim=1).cpu().numpy(),
            "condition": attr_logits["condition"].argmax(dim=1).cpu().numpy(),
            "size": attr_logits["size"].argmax(dim=1).cpu().numpy(),
        }

        batch_size = imgs.size(0)

        for i in range(batch_size):
            # Get the original sample from the dataset to read its row metadata
            sample = loader.dataset[dataset_idx + i]
            row = sample[-1]   # last element should be row / metadata

            # --- existing path logic, but using row from dataset ---
            if isinstance(row, (list, tuple)):
                img_path = row[0]  # first field is image path
            elif isinstance(row, dict):
                img_path = row.get("image_path", "")
            else:
                img_path = str(row)

            # Build an absolute path for the image
            if os.path.isabs(img_path):
                img_abs = img_path
            else:
                img_abs = os.path.abspath(os.path.join(DATA_DIR, img_path))

            # Normalize slashes
            img_abs = img_abs.replace("\\", "/")

            # DEBUG: print a couple of paths to verify they are real images
            if len(index) < 3:
                print("INDEX SAMPLE PATH:", img_abs)

            # Create a URL-safe token from the absolute path
            img_token = base64.urlsafe_b64encode(img_abs.encode("utf-8")).decode("ascii")

            index.append({
                "feat": feats[i].cpu().numpy(),
                "pred_class": int(preds_class[i]),
                "pred_color": int(preds_attr["color"][i]),
                "pred_material": int(preds_attr["material"][i]),
                "pred_condition": int(preds_attr["condition"][i]),
                "pred_size": int(preds_attr["size"][i]),
                "image_abspath": img_abs,
                "image_token": img_token,
            })

        dataset_idx += batch_size

    return index


# -----------------------------
# PREDICTION + RETRIEVAL LOGIC
# -----------------------------

def predict_image(bundle: ModelBundle, image_path: str):
    """
    Run class + attribute prediction on a single image file path.
    """
    img = Image.open(image_path).convert("RGB")
    x = IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats, class_logits, attr_logits = bundle.model(x)

    class_idx = int(class_logits.argmax(dim=1).item())
    class_name = bundle.idx_to_class[class_idx]

    attrs_pred = {}
    for name in ATTR_NAMES:
        idx = int(attr_logits[name].argmax(dim=1).item())
        attrs_pred[name] = bundle.attr_schema[name][idx]

    return class_name, attrs_pred


def parse_query(query: str, bundle: ModelBundle):
    """
    Parse a query like:
        "class=jug color=blue material=plastic size=small"
    or free text like:
        "blue plastic jug"
    into target class index + attribute indices (or None if unspecified).
    """
    q = query.lower()
    tokens = q.split()

    result = {
        "class": None,
        "color": None,
        "material": None,
        "condition": None,
        "size": None,
    }

    found_kv = False
    for tok in tokens:
        if "=" in tok:
            key, value = tok.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "class":
                for cls_name, idx in bundle.class_to_idx.items():
                    if value in cls_name.lower():
                        result["class"] = idx
                        break
            elif key in bundle.attr_schema:
                for i, v in enumerate(bundle.attr_schema[key]):
                    if value == v.lower():
                        result[key] = i
                        break
            found_kv = True

    if found_kv:
        return result

    # Free-text mode
    qtext = " " + q + " "

    # class
    for cls_name, idx in bundle.class_to_idx.items():
        if cls_name.lower() in qtext:
            result["class"] = idx
            break

    # attributes
    for attr in ATTR_NAMES:
        for i, v in enumerate(bundle.attr_schema[attr]):
            if v == "unknown":
                continue
            if f" {v.lower()} " in qtext:
                result[attr] = i
                break

    return result


def retrieve_images(bundle: ModelBundle, query: str, top_k: int = 5):
    q_dict = parse_query(query, bundle)
    index = bundle.index

    scores = []

    for item in index:
        score = 0.0

        qc = q_dict["class"]
        if qc is not None and item["pred_class"] == qc:
            score += 1.0

        for attr in ["color", "material", "condition", "size"]:
            qv = q_dict[attr]
            if qv is not None and item[f"pred_{attr}"] == qv:
                score += 0.5

        scores.append(score)

    scores = np.array(scores)
    order = np.argsort(scores)[::-1]

    results = []
    for idx in order:
        if scores[idx] <= 0:
            continue

        item = index[idx]
        img_token = item["image_token"]

        cls_name = bundle.idx_to_class[item["pred_class"]]
        attrs = {}
        for a in ATTR_NAMES:
            av_idx = item[f"pred_{a}"]
            attrs[a] = bundle.attr_schema[a][av_idx]

        results.append({
            "image_token": img_token,
            "class_name": cls_name,
            "attributes": attrs,
        })

        if len(results) >= top_k:
            break

    return results


# -----------------------------
# FLASK APP
# -----------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_options=["deit", "vit"],
        dataset_options=["ours", "pooled"],
        default_top_k=5,
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    model_name = request.form.get("model_name", "deit")
    dataset_variant = request.form.get("dataset_variant", "ours")

    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    bundle = load_model_bundle(model_name, dataset_variant)
    class_name, attrs_pred = predict_image(bundle, save_path)

    return render_template(
        "index.html",
        model_options=["deit", "vit"],
        dataset_options=["ours", "pooled"],
        default_top_k=5,
        pred_image=url_for("uploaded_file", filename=filename),
        pred_model=model_name,
        pred_dataset=dataset_variant,
        pred_class=class_name,
        pred_attrs=attrs_pred,
    )


@app.route("/retrieve", methods=["POST"])
def retrieve_route():
    model_name = request.form.get("model_name", "deit")
    dataset_variant = request.form.get("dataset_variant", "ours")
    query = request.form.get("query", "")
    try:
        top_k = int(request.form.get("top_k", "5"))
    except ValueError:
        top_k = 5

    if not query.strip():
        return redirect(url_for("index"))

    bundle = load_model_bundle(model_name, dataset_variant)
    results = retrieve_images(bundle, query, top_k=top_k)

    # augment results with URL to serve images
    for r in results:
        r["url"] = url_for("serve_data_image", path=r["image_token"])

    return render_template(
        "index.html",
        model_options=["deit", "vit"],
        dataset_options=["ours", "pooled"],
        default_top_k=top_k,
        retrieval_query=query,
        retrieval_model=model_name,
        retrieval_dataset=dataset_variant,
        retrieval_results=results,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/data/<path:path>")
def serve_data_image(path):
    """Serve an image used in the dataset.

    `path` is a URL-safe base64 token encoding the absolute file path.
    """
    from flask import abort

    # DEBUG: see what token we got
    print(">>> /data got token:", path)

    try:
        full_path = base64.urlsafe_b64decode(path.encode("ascii")).decode("utf-8")
    except Exception as e:
        print("!!! decode error:", e)
        return abort(400)

    # Normalize slashes
    full_path = full_path.replace("\\", "/")

    # DEBUG: see what file path we resolved to and whether it exists
    print(">>> decoded full_path:", full_path, "exists:", os.path.exists(full_path))

    if not os.path.exists(full_path):
        return abort(404)

    return send_file(full_path)

if __name__ == "__main__":
    app.run(debug=True)
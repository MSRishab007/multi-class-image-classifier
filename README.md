# Multi-Class Image Classifier

This project is a multi-task image classification and retrieval demo built with PyTorch, Flask, and timm. It supports two dataset variants, two backbone choices, class prediction, attribute prediction, and text-driven image retrieval.

The active frontend lives in `web_app1/`. The legacy `web_app/` folder now forwards to the same app for compatibility.

For Hugging Face Spaces, the repo root now includes a lightweight `app.py` that launches the same Flask app directly.

## What It Does

- Predicts the object class for an uploaded image.
- Predicts object attributes: color, material, condition, and size.
- Retrieves visually similar images using text queries.
- Supports fuzzy class matching, so misspellings like `keywoard` still try to match the closest class.
- Supports multiple values per attribute, so queries like `color=red,blue` match either value.

## Project Layout

- `web_app1/app.py`: active Flask application.
- `app.py`: root launcher for Hugging Face Spaces and other root-level app runners.
- `web_app1/templates/index.html`: prediction and retrieval UI.
- `web_app/app.py`: compatibility wrapper that forwards to `web_app1`.
- `src/dataset.py`: dataset loader for the original CSV format.
- `src/dataset_pooled.py`: dataset loader for the pooled CSV format.
- `src/model.py`: multi-task model definition used by the active app.
- `src/train_*.py`: training scripts for DeiT and ViT, original and pooled variants.
- `src/retrieval.py`: standalone retrieval script.
- `data/`: labels, attributes, and images used by the app.
- `outputs*/`: saved model weights.

## Dataset Variants

The app supports two dataset setups:

- `ours`: uses `data/labels.csv` and `data/attributes.yaml`.
- `pooled`: uses `data/processed_dataset.csv` and `data/attributes_pooled.yaml`.

The retrieval filters are generated from the attribute schema, so the options always match the active dataset.

## Models

Supported backbones:

- `deit`: DeiT-Tiny.
- `vit`: ViT-Tiny.

Each backbone has weights for both dataset variants under the `outputs/` folders.

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

4. Open the local URL printed by Flask, usually `http://127.0.0.1:7860`.

## Testing

Run the built-in smoke tests with:

```bash
python -m unittest discover -s tests
```

The smoke suite checks the homepage, prediction route, retrieval route, and dataset image serving.

## Query Examples

The retrieval UI accepts free text and explicit key-value filters.

- `keyboard`
- `keywoard`
- `class=keyboard color=red,blue`
- `class=cup material=plastic condition=new`

Multiple values in one attribute are treated as OR conditions. For example, `color=red,blue` matches items predicted as either red or blue.

## Hugging Face Deployment

This repository is ready to deploy as a Python app on Hugging Face Spaces.

Recommended setup:

1. Create a Python Space.
2. Push this repository to the Space.
3. Make sure `requirements.txt` is included.
4. Launch the app with:

```bash
python app.py
```

The app reads `HOST` and `PORT` from the environment, so it works with Spaces-style deployment settings. The default port is `7860` and the host is `0.0.0.0`.

## Notes

- The app serves dataset images directly from the `data/` folder.
- Uploaded images are stored in `uploads/`.
- If a weight file is missing, check the relevant `outputs*/best_model.pth` path.
- If you want to retrain the models, use the scripts in `src/train_*.py`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

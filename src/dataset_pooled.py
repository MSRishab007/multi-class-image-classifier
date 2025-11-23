import os
import csv
import yaml
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class EverydayObjectsDataset(Dataset):
    def __init__(self, root, split, attrs_yaml="attributes_pooled.yaml",
                 labels_csv="processed_dataset.csv", img_size=224):
        """
        root: path to project/data (folder containing images/, labels.csv, attributes.yaml)
        split: "train" or "val"
        """
        self.root = root
        self.split = split
        
        # Load attributes schema
        with open(os.path.join(root, attrs_yaml)) as f:
            self.attr_schema = yaml.safe_load(f)
        
        # Build value -> idx maps
        self.attr_value_to_idx = {}
        for attr_name, values in self.attr_schema.items():
            self.attr_value_to_idx[attr_name] = {v: i for i, v in enumerate(values)}
        
        # Load CSV
        self.samples = []
        with open(os.path.join(root, labels_csv), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                self.samples.append(row)
        
        # Class mapping
        class_labels = sorted({s["mapped_class"] for s in self.samples})
        self.class_to_idx = {c: i for i, c in enumerate(class_labels)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        
        # Transforms (resize from 96x96 to 224x224 for ViT)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def parse_attributes(self, attr_str):
        """
        'color:black;material:plastic;condition:new;size:small'
        -> dict { 'color': idx, 'material': idx, ... }
        """
        result = {}
        for kv in attr_str.split(";"):
            if not kv:
                continue
            parts = kv.split(":", 1)  # Split on first colon only
            if len(parts) != 2:
                continue
            key, value = parts
            key = key.strip()
            value = value.strip()
            idx = self.attr_value_to_idx[key][value]
            result[key] = idx
        return result
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        img_path = os.path.join(self.root, row["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        # Class label
        y_class = self.class_to_idx[row["mapped_class"]]
        
        # Attributes
        attr_indices = self.parse_attributes(row["attributes"])
        y_color = attr_indices["color"]
        y_material = attr_indices["material"]
        y_condition = attr_indices["condition"]
        y_size = attr_indices["size"]
        
        return (
            img,
            torch.tensor(y_class, dtype=torch.long),
            torch.tensor(y_color, dtype=torch.long),
            torch.tensor(y_material, dtype=torch.long),
            torch.tensor(y_condition, dtype=torch.long),
            torch.tensor(y_size, dtype=torch.long),
            row  # for bookkeeping (caption, instance_id, etc.)
        )

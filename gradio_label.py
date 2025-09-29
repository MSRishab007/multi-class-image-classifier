import os
import re
import yaml
import csv
import gradio as gr
from collections import defaultdict

# --- Load schema ---
with open("attributes.yaml", "r") as f:
    schema = yaml.safe_load(f)

# Now include size as well
attributes = ["color", "material", "condition", "size"]

# --- Config ---
input_folder = r"C:\Users\Nimish\Downloads\DL_project\images"
output_csv = "final_labels.csv"

files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
files.sort()
total_images = len(files)
index = 0
rows = []

split_counts = defaultdict(lambda: {"train": 0, "val": 0})
instance_split = {}

def parse_filename(fname):
    match = re.match(r"team9_([a-z]+)_(\d+)_([a-z])\.(jpg|jpeg|png)$", fname, re.IGNORECASE)
    if match:
        class_label, inst_num, letter, ext = match.groups()
        instance_id = f"{class_label}_{inst_num}"
        return class_label, instance_id
    return "unknown", "unknown"

def get_image(idx):
    if idx < len(files):
        filepath = os.path.join(input_folder, files[idx])
        return filepath, files[idx]
    return None, "Done"

def submit_label(color, material, condition, size, caption, split):
    global index, rows, split_counts, instance_split

    if index < len(files):
        filename = files[index]
        image_path = os.path.join(input_folder, filename)
        class_label, instance_id = parse_filename(filename)

        # --- Enforce instance consistency ---
        if instance_id in instance_split:
            split = instance_split[instance_id]
        else:
            total_class = split_counts[class_label]["train"] + split_counts[class_label]["val"]
            if split == "val":
                if total_class > 0 and split_counts[class_label]["val"] / total_class >= 0.2:
                    split = "train"
            instance_split[instance_id] = split

        # Build attributes string (now includes size)
        attr_string = f"color:{color};material:{material};condition:{condition};size:{size}"

        rows.append([image_path, class_label, attr_string, caption, split, instance_id])
        split_counts[class_label][split] += 1
        index += 1

        if index == len(files):
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image_path", "class_label", "attributes", "caption", "split", "instance_id"])
                writer.writerows(rows)
            return None, f"✅ All {total_images} images labeled. Saved to {output_csv}"

    filepath, fname = get_image(index)
    progress = f"Image {index+1} of {total_images} | " + \
               " | ".join([f"{cls}: train={c['train']}, val={c['val']}" for cls,c in split_counts.items()])
    return filepath, f"Labeling: {fname} | {progress}"

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🏷️ Image Labeling Tool with Size Attribute")

    image = gr.Image(type="filepath", label="Image to Label")
    status = gr.Label(label="Progress / Status")

    with gr.Row():
        color_dd = gr.Dropdown(schema["color"], label="Color", value="unknown")
        material_dd = gr.Dropdown(schema["material"], label="Material", value="unknown")
        condition_dd = gr.Dropdown(schema["condition"], label="Condition", value="unknown")
        size_dd = gr.Dropdown(schema["size"], label="Size", value="unknown")

    caption_box = gr.Textbox(label="Caption", placeholder="Type a caption for this image...")
    split_dd = gr.Dropdown(["train", "val"], label="Split", value="train")

    submit_btn = gr.Button("Submit & Next")

    submit_btn.click(
        fn=submit_label,
        inputs=[color_dd, material_dd, condition_dd, size_dd, caption_box, split_dd],
        outputs=[image, status]
    )

    first_path, first_name = get_image(index)
    image.value = first_path
    status.value = f"Labeling: {first_name} | Image 1 of {total_images}"

demo.launch()

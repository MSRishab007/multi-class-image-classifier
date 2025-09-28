from PIL import Image
import os

# Input and output folders
input_folder = r"C:\Users\Nimish\Downloads\DL_project\input_images"
output_folder = r"C:\Users\Nimish\Downloads\DL_project\output_images"

os.makedirs(output_folder, exist_ok=True)

# Target size
target_size = (96, 96)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")  # ensure RGB for JPEG

        # Step 1: Resize proportionally
        img.thumbnail(target_size, Image.LANCZOS)

        # Step 2: Create a new square canvas (white background)
        new_img = Image.new("RGB", target_size, (255, 255, 255))

        # Step 3: Paste resized image centered
        paste_x = (target_size[0] - img.size[0]) // 2
        paste_y = (target_size[1] - img.size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))

        # Save as JPEG
        base_name = os.path.splitext(filename)[0] + ".jpg"
        new_img.save(os.path.join(output_folder, base_name), "JPEG", quality=95)

print("✅ All images resized and saved as 96x96 JPGs.")

import os

input_folder = r"C:\Users\Nimish\Downloads\DL_project\output_images"

for filename in os.listdir(input_folder):
    # Only process image files
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        old_path = os.path.join(input_folder, filename)

        # Replace hyphens with underscores
        new_filename = filename.replace("-", "_")
        new_path = os.path.join(input_folder, new_filename)

        # Rename only if different
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

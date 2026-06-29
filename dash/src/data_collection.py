"""
Google Street View Image Collection and Preprocessing
Handles mounting drives and initial image organization
"""

from google.colab import drive
import os

def setup_project_environment():
    """Mount Google Drive and verify project structure"""
    drive.mount('/content/drive')

    project_path = "/content/drive/MyDrive/247C Project/data/standardized_inventory"

    if os.path.exists(project_path):
        print(f"✅ Success! Found {len(os.listdir(project_path))} cropped images.")
        return project_path
    else:
        print("❌ Error: Path not found. Check your folder names in Google Drive.")
        return None

def verify_image_paths(path_to_check):
    """Walk through directories to find actual image files"""
    print(f"Checking: {path_to_check}")

    found_images = []
    for root, dirs, files in os.walk(path_to_check):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                found_images.append(os.path.join(root, file))

    if not found_images:
        print("❌ Still 0 images. Are you sure the Drive is mounted and files uploaded?")
        return None
    else:
        print(f"✅ Found {len(found_images)} images!")
        print(f"First image path: {found_images[0]}")
        return found_images

if __name__ == "__main__":
    setup_project_environment()

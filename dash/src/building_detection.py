"""
Building Detection Pipeline using RoboFlow and GroundingDINO
Crops buildings from street view images
"""

from inference_sdk import InferenceHTTPClient
import os
from PIL import Image
from tqdm import tqdm
import shutil

class BuildingDetector:
    def __init__(self, api_key="983KfTAJW9jf6Ww5T91o"):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
    
    def resize_with_padding(self, img, expected_size=(640, 640)):
        """Resize image with padding to maintain aspect ratio"""
        img.thumbnail(expected_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", expected_size, (0, 0, 0))
        x_offset = (expected_size[0] - img.width) // 2
        y_offset = (expected_size[1] - img.height) // 2
        canvas.paste(img, (x_offset, y_offset))
        return canvas
    
    def detect_and_crop_buildings(self, input_folder, output_folder, label_prefix):
        """Process images through RoboFlow building detection workflow"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        images = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"🚀 Processing {len(images)} images, selecting largest facades...")
        
        for img_name in tqdm(images):
            input_path = os.path.join(input_folder, img_name)
            try:
                # Run RoboFlow workflow
                result = self.client.run_workflow(
                    workspace_name="buildings-e5th4",
                    workflow_id="find-buildings",
                    images={"image": input_path}
                )
                
                if result and 'predictions' in result[0]:
                    preds = result[0]['predictions']
                    object_list = preds if isinstance(preds, list) else preds.get('predictions', [])
                    
                    if object_list:
                        # Select largest detection
                        best_pred = max(object_list, key=lambda x: x['width'] * x['height'])
                        
                        with Image.open(input_path) as img:
                            # Extract bounding box coordinates
                            x_c, y_c, w, h = best_pred['x'], best_pred['y'], best_pred['width'], best_pred['height']
                            left, top = x_c-(w/2), y_c-(h/2)
                            right, bottom = x_c+(w/2), y_c+(h/2)
                            
                            # Crop and standardize
                            crop = img.crop((left, top, right, bottom))
                            standardized_crop = self.resize_with_padding(crop)
                            
                            # Save with consistent naming
                            crop_filename = f"{label_prefix}.{img_name}"
                            standardized_crop.save(os.path.join(output_folder, crop_filename))
                            
            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")
                continue
        
        print(f"✅ Finished! Clean inventory ready in: {output_folder}")
    
    def organize_primary_facades(self, inventory_folder, labeled_folder):
        """Extract primary building views and organize with consistent labeling"""
        existing_files = os.listdir(labeled_folder)
        existing_ids = {f.split('.')[-1].replace('.jpg', '') for f in existing_files if '.' in f}
        
        inventory_files = [f for f in os.listdir(inventory_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"🔍 Organizing {len(inventory_files)} inventory files...")
        
        moved_count = 0
        for f in tqdm(inventory_files):
            if "NON_SOFT_STORY" in f and "_0_" in f:
                parts = f.split('_0_')
                if len(parts) > 1:
                    img_id = parts[1].replace('.jpg', '')
                    
                    if img_id not in existing_ids:
                        new_name = f"non_soft_story.{img_id}.jpg"
                        src_path = os.path.join(inventory_folder, f)
                        dst_path = os.path.join(labeled_folder, new_name)
                        shutil.copy(src_path, dst_path)
                        moved_count += 1
        
        print(f"✅ Added {moved_count} new primary facades to labeled folder.")

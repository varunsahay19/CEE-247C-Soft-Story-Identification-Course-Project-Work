"""
DINO Feature Extraction Pipeline
Extracts 384-dimensional embeddings from cropped building images
"""

import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm

class DinoFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Load pre-trained DINO v2 model
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dinov2.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def create_label_csv(self, folder_path, output_csv):
        """Generate CSV with filenames and soft-story labels"""
        all_files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        data = []
        print(f"🚀 Generating CSV for {len(all_files)} images...")
        
        for filename in tqdm(all_files):
            fname_lower = filename.lower()
            
            if "non_soft_story" in fname_lower:
                label = 0.0
            elif "soft_story" in fname_lower:
                label = 1.0
            else:
                continue  # Skip unlabeled files
            
            data.append({'filename': filename, 'label': label})
        
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        
        print(f"✅ CSV saved to: {output_csv}")
        print(f"📊 Label distribution:\n{df['label'].value_counts()}")
        return df
    
    def extract_features(self, df_labels, img_dir, output_csv):
        """Extract DINO features for all labeled images"""
        all_features = []
        print("🧬 Extracting DINO embeddings...")
        
        for fname in tqdm(df_labels['filename']):
            img_path = os.path.join(img_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.dinov2(img_tensor).cpu().numpy().flatten()
                all_features.append(features)
                
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                all_features.append([0] * 384)  # Fallback zero vector
        
        # Create feature dataframe
        feature_cols = [f'feat_{i}' for i in range(384)]
        feat_df = pd.DataFrame(all_features, columns=feature_cols)
        
        # Combine with labels
        final_df = pd.concat([df_labels, feat_df], axis=1)
        final_df.to_csv(output_csv, index=False)
        
        print(f"✅ Features extracted and saved to: {output_csv}")
        return final_df

# Usage example
if __name__ == "__main__":
    extractor = DinoFeatureExtractor()
    
    # Create labels CSV
    folder_path = "/content/drive/MyDrive/247C Project/data/standardized_images_dash_labeled"
    label_csv = "/content/drive/MyDrive/247C Project/data/standardized_images_dash_labeled.csv"
    df_labels = extractor.create_label_csv(folder_path, label_csv)
    
    # Extract features
    features_csv = "/content/drive/MyDrive/247C Project/data/final_training_data_with_features_dash_labeled.csv"
    final_df = extractor.extract_features(df_labels, folder_path, features_csv)

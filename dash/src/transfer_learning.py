"""
Transfer Learning Validation on Los Angeles Buildings
Tests model generalization across cities
"""

import torch
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

class TransferLearningValidator:
    def __init__(self, model, dinov2, scaler, transform, device=None):
        self.model = model
        self.dinov2 = dinov2
        self.scaler = scaler
        self.transform = transform
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set models to evaluation mode
        self.model.to(self.device).eval()
        self.dinov2.to(self.device).eval()
    
    def predict_la_buildings(self, input_dir, output_csv):
        """Predict soft-story status for Los Angeles buildings"""
        print(f"🧐 Analyzing LA buildings from: {input_dir}")
        
        images = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        results = []
        
        for fname in tqdm(images, desc="Processing LA buildings"):
            img_path = os.path.join(input_dir, fname)
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Extract DINO features
                    features = self.dinov2(img_tensor)
                    
                    # Generate prediction
                    prediction_prob = self.model(features).item()
                    predicted_class = 1 if prediction_prob > 0.5 else 0
                    
                    results.append({
                        'filename': fname,
                        'soft_story_probability': f"{prediction_prob:.4f}",
                        'predicted_label': predicted_class,
                        'confidence': 'High' if abs(prediction_prob - 0.5) > 0.3 else 'Medium'
                    })
                    
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")
                results.append({
                    'filename': fname,
                    'soft_story_probability': 'ERROR',
                    'predicted_label': -1,
                    'confidence': 'Failed'
                })
        
        # Save results
        prediction_df = pd.DataFrame(results)
        prediction_df.to_csv(output_csv, index=False)
        
        print(f"\n✅ LA Predictions Complete!")
        print(f"📊 Processed {len(prediction_df)} buildings")
        print(f"💾 Results saved to: {output_csv}")
        
        # Summary statistics
        valid_predictions = prediction_df[prediction_df['predicted_label'] != -1]
        if len(valid_predictions) > 0:
            soft_story_count = (valid_predictions['predicted_label'] == 1).sum()
            print(f"🏗️ Predicted soft-story buildings: {soft_story_count}/{len(valid_predictions)}")
            print(f"📈 Soft-story rate: {soft_story_count/len(valid_predictions)*100:.1f}%")
        
        return prediction_df
    
    def analyze_transfer_performance(self, la_predictions):
        """Analyze the quality of transfer learning predictions"""
        valid_preds = la_predictions[la_predictions['predicted_label'] != -1]
        
        if len(valid_preds) == 0:
            print("❌ No valid predictions to analyze")
            return
        
        print("\n🔍 TRANSFER LEARNING ANALYSIS")
        print("=" * 40)
        
        # Confidence distribution
        confidence_dist = valid_preds['confidence'].value_counts()
        print(f"Confidence Distribution:")
        for conf, count in confidence_dist.items():
            print(f"  {conf}: {count} buildings ({count/len(valid_preds)*100:.1f}%)")
        
        # Probability distribution
        probs = pd.to_numeric(valid_preds['soft_story_probability'], errors='coerce')
        print(f"\nPrediction Statistics:")
        print(f"  Mean probability: {probs.mean():.3f}")
        print(f"  Std deviation: {probs.std():.3f}")
        print

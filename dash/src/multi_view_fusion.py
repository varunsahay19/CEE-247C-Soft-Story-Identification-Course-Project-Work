"""
Multi-View Building Classification
Aggregates predictions from multiple street views of the same building
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score

class MultiViewFusion:
    def __init__(self, model, scaler, device=None):
        self.model = model
        self.scaler = scaler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def extract_building_id(self, filename):
        """Extract building ID from filename"""
        return filename.split('_view')[0]
    
    def predict_all_views(self, df):
        """Generate predictions for all individual views"""
        X_all = df.filter(like='feat_').values
        X_all_tensor = torch.FloatTensor(self.scaler.transform(X_all)).to(self.device)
        
        with torch.no_grad():
            probabilities = self.model(X_all_tensor).cpu().numpy().flatten()
        
        df = df.copy()
        df['building_id'] = df['filename'].apply(self.extract_building_id)
        df['soft_story_probability'] = probabilities
        df['photo_vote'] = (probabilities > 0.5).astype(int)
        
        return df
    
    def aggregate_building_predictions(self, df_with_predictions):
        """Aggregate view-level predictions to building level"""
        building_level = df_with_predictions.groupby('building_id').agg({
            'soft_story_probability': ['max', 'mean'],
            'photo_vote': 'mean',
            'label': 'max'  # Building label (same for all views)
        }).reset_index()
        
        # Flatten column names
        building_level.columns = [
            'building_id', 'max_prob', 'avg_prob', 'vote_count', 'actual_label'
        ]
        
        # Apply fusion rules
        building_level['Any_Soft_Rule'] = (building_level['max_prob'] > 0.5).astype(int)
        building_level['Average_Rule'] = (building_level['avg_prob'] > 0.5).astype(int)
        building_level['Most_Frequent_Rule'] = (building_level['vote_count'] > 0.5).astype(int)
        
        return building_level
    
    def evaluate_fusion_strategies(self, building_level_df):
        """Compare performance of different fusion strategies"""
        y_true = building_level_df['actual_label']
        
        strategies = {
            'Any_Soft_Rule': building_level_df['Any_Soft_Rule'],
            'Average_Rule': building_level_df['Average_Rule'], 
            'Most_Frequent_Rule': building_level_df['Most_Frequent_Rule']
        }
        
        results = {}
        print("📊 MULTI-VIEW FUSION PERFORMANCE")
        print("=" * 50)
        
        for strategy_name, y_pred in strategies.items():
            accuracy = (y_pred == y_true).mean()
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            
            results[strategy_name] = {
                'accuracy': accuracy,
                'recall': recall,
                'f1': f1,
                'precision': precision
            }
            
            print(f"\n--- {strategy_name} ---")
            print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"Recall:    {recall:.4f} (Caught {recall*100:.1f}% of risky buildings)")
            print(f"F1 Score:  {f1:.4f}")
            print(f"Precision: {precision:.4f}")
        
        return results
    
    def analyze_test_set_performance(self, df, test_size=0.2):
        """Evaluate fusion performance on unseen buildings"""
        from sklearn.model_selection import train_test_split
        
        # Get unique buildings for proper train/test split
        unique_buildings = df['building_id'].unique()
        train_buildings, test_buildings = train_test_split(
            unique_buildings, test_size=test_size, random_state=42
        )
        
        # Filter to test buildings only
        test_df = df[df['building_id'].isin(test_buildings)].copy()
        
        # Apply predictions to test set
        test_df = self.predict_all_views(test_df)
        test_building_level = self.aggregate_building_predictions(test_df)
        
        print(f"🕵️‍♂️ UNSEEN BUILDING PERFORMANCE (Test Set)")
        print(f"Total unique test buildings: {len(test_building_level)}")
        print("-" * 50)
        
        return self.evaluate_fusion_strategies(test_building_level)

# Usage example
if __name__ == "__main__":
    # Load trained model and data
    model = SoftStoryClassifier()
    model.load_state_dict(torch.load("path/to/model.pth"))
    
    # Initialize fusion analyzer
    fusion = MultiViewFusion(model, scaler)  # scaler from training
    
    # Analyze performance
    df = pd.read_csv("path/to/features.csv")
    results = fusion.analyze_test_set_performance(df)

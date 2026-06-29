"""
Soft-Story Classification Model Training
Neural network training on DINO features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class SoftStoryClassifier(nn.Module):
    """Neural network for soft-story building classification"""
    
    def __init__(self, input_dim=384):
        super(SoftStoryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class ModelTrainer:
    def __init__(self, model_save_path):
        self.model = SoftStoryClassifier()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = StandardScaler()
        self.model_save_path = model_save_path
    
    def prepare_data(self, df):
        """Prepare training and testing data"""
        X = df.filter(like='feat_').values
        y = df['label'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalize features
        X_train = torch.FloatTensor(self.scaler.fit_transform(X_train))
        X_test = torch.FloatTensor(self.scaler.transform(X_test))
        y_train = torch.FloatTensor(y_train).view(-1, 1)
        y_test = torch.FloatTensor(y_test).view(-1, 1)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, epochs=75):
        """Train the soft-story classification model"""
        print("🚀 Training Soft-Story Predictor...")
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Save trained model
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"💾 Model saved to: {self.model_save_path}")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance with detailed metrics"""
        self.model.eval()
        with torch.no_grad():
            raw_predictions = self.model(X_test)
            y_pred = (raw_predictions > 0.5).float().cpu().numpy()
            y_true = y_test.cpu().numpy()
        
        # Calculate metrics
        accuracy = (y_pred == y_true).mean()
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"✅ Training Complete!")
        print(f"📊 Performance Metrics:")
        print(f"   - Accuracy: {accuracy*100:.2f}%")
        print(f"   - Recall (Risk Detection): {recall*100:.2f}%")
        print(f"   - F1 Score: {f1*100:.2f}%")
        
        # Detailed classification report
        print("\n📝 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Safe', 'Soft-Story']))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        return accuracy, recall, f1
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Safe', 'Predicted Risk'],
                    yticklabels=['Actual Safe', 'Actual Risk'])
        plt.title('Soft-Story Confusion Matrix')
        plt.show()

# Usage example
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("/content/drive/MyDrive/247C Project/data/final_training_data_with_features_dash_labeled.csv")
    
    # Initialize trainer
    trainer = ModelTrainer("/content/drive/MyDrive/247C Project/data/soft_story_model.pth")
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Train model
    trainer.train_model(X_train, y_train)
    
    # Evaluate model
    trainer.evaluate_model(X_test, y_test)

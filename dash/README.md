# Seismic Building Detection: Vision-Based Soft-Story Identification

Computer vision system for automated soft-story building identification using DINO embeddings and multi-view fusion. Developed to support San Francisco's seismic retrofit prioritization program.

## Overview

Dash Beavers's part of this project develops an AI pipeline that automatically identifies seismically vulnerable soft-story buildings from Google Street View imagery. The system combines state-of-the-art computer vision models (GroundingDINO, DINO v2) with custom classification networks to reduce manual building inspections by 51.3% while maintaining 82.7% recall for safety-critical applications.

![Pipeline Architecture](dash/results/pipeline_architecture.png)

## Key Results

- **82.9% accuracy** and **82.7% recall** on San Francisco building dataset
- **51.3% reduction** in required manual building surveys
- **Successful transfer learning** to Los Angeles buildings
- **Multi-view fusion** strategies for improved building-level predictions

![Workflow Demonstration](dash/results/workflow_demonstration.png)

## Technical Approach

### 1. Building Detection & Preprocessing
- **GroundingDINO + RoboFlow** for automated building localization
- **Image standardization** with padding-based resizing
- **Multi-view collection** from Google Street View API

### 2. Feature Extraction
- **DINO v2** (ViT-S/14) for 384-dimensional semantic embeddings
- **Pretrained representations** adapted for structural analysis
- **Robust to occlusions** and varying image quality

### 3. Classification Network
```python
class SoftStoryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```

### 4. Multi-View Fusion
- **Any-Soft Rule**: Building flagged if ANY view indicates soft-story
- **Majority Rule**: Building flagged if >50% of views indicate soft-story
- **Conservative safety-oriented** approach prioritizing recall

## Dataset
- **8,209 buildings** from San Francisco's Mandatory Seismic Retrofit Program
- **~18,000 street-view images** across multiple viewing angles
- **4,945 soft-story** and **3,264 non-soft-story** buildings
- **Ground truth labels** from engineering surveys and city records

## Installation
```python
# Clone repository
git clone https://github.com/dashbeavers/seismic-building-detection.git
cd seismic-building-detection

# Install dependencies
pip install -r requirements.txt

# Install additional requirements for DINO
pip install torch torchvision
pip install transformers==4.29.2
```

## Usage

### 1. Data Collection
```python
from src.data_collection import setup_project_environment
setup_project_environment()
```

### 2. Building Detection
```python
from src.building_detection import BuildingDetector

detector = BuildingDetector()
detector.detect_and_crop_buildings(input_folder, output_folder, "soft_story")
```

### 3. Feature Extraction
```python
from src.feature_extraction import DinoFeatureExtractor

extractor = DinoFeatureExtractor()
features_df = extractor.extract_features(df_labels, img_dir, output_csv)
```

### 4. Model Training
```python
from src.model_training import ModelTrainer

trainer = ModelTrainer("model.pth")
X_train, X_test, y_train, y_test = trainer.prepare_data(df)
trainer.train_model(X_train, y_train)
trainer.evaluate_model(X_test, y_test)
```

### 5. Multi-View Analysis
```python
from src.multi_view_fusion import MultiViewFusion

fusion = MultiViewFusion(model, scaler)
results = fusion.analyze_test_set_performance(df)
```

## Transfer Learning Results

Successful validation on Los Angeles buildings demonstrates cross-city generalization:

![LA Validation Examples](dash/results/la_validation_example.png)

The model correctly identifies structural characteristics across different:
- **Architectural styles** (SF Victorian vs LA Commercial)
- **Urban contexts** (dense vs suburban environments) 
- **Image conditions** (lighting, occlusion, perspective)

## Performance Metrics

| Fusion Strategy | Accuracy | Recall | F1 Score |
|----------------|----------|---------|----------|
| Any-Soft Rule | 82.9% | 82.7% | 82.4% |
| Majority Rule | 83.5% | 79.1% | 82.3% |

**Key Insight**: Any-Soft Rule prioritizes safety by maximizing recall (fewer missed vulnerable buildings), while Majority Rule improves precision by reducing false alarms.

## Project Structure
```python
seismic-building-detection/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── src/
│   ├── data_collection.py         # Google Drive setup and image organization
│   ├── building_detection.py      # RoboFlow + GroundingDINO pipeline
│   ├── feature_extraction.py      # DINO v2 embedding extraction
│   ├── model_training.py          # Neural network training and evaluation
│   ├── multi_view_fusion.py       # Building-level prediction aggregation
│   └── transfer_learning.py       # Cross-city validation (LA buildings)
├── results/
│   ├── pipeline_architecture.png  # Technical workflow visualization
│   ├── workflow_demonstration.png # Building detection process
│   ├── results_visualized.png    # Model evaluation results
│   └── la_validation_example.png # Transfer learning examples
└── technical_report.pdf          # Complete methodology and results
```

## Technologies Used

- **Computer Vision**: GroundingDINO, DINO v2, OpenCV
- **Deep Learning**: PyTorch, torchvision, transformers
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, folium
- **APIs**: Google Street View, RoboFlow
- **Development**: Python, Jupyter, Google Colab

## Real-World Impact

This system addresses a critical public safety challenge by:

- **Accelerating seismic assessments** from months to days
- **Reducing inspection costs** by focusing efforts on high-risk buildings
- **Supporting equitable retrofits** through data-driven prioritization
- **Enabling city-wide coverage** previously impractical with manual surveys

The methodology is designed to support, not replace, engineering judgment - serving as a screening tool that prioritizes buildings for professional inspection.

## Future Work

- **Greatly expand to additional cities** (Oakland, Berkeley, Los Angeles)
- **Integration with permit records** and building age databases
- **Real-time monitoring** of new construction and renovations
- **Mobile deployment** for field inspection support
- **Interpretability analysis** using attention maps and feature visualization

## Technical Report

For complete methodology, evaluation details, and literature review, see the [full technical report](technical_report.pdf).

## Citation

If you use this work in your research, please cite:

```bibtex
@article{beavers2026seismic,
  title={Vision-Based Identification of Soft-Story Buildings for Seismic Retrofit Prioritization},
  author={Beavers, Dash and Sahay, Varun and Nguyen, Kim and Zou, Yushun},
  journal={Computer Vision for the Built Environment},
  year={2026}
}
```
## Contributors

- **Dash Beavers** 
- **Varun Sahay**
- **Kim Nguyen**
- **Yushun Zou**

*Group project for Computer Vision for the Built Environment (Stanford University)*

## Contact

**Dash Beavers**  
📧 [dashbeavers@gmail.com](mailto:dashbeavers@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/dashbeavers)  
🐙 [GitHub](https://github.com/dashbeavers)

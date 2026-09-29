# Bharatanatyam Mudra Recognition - CNN Classification

## Project Overview
This project implements a **5-class hand gesture image classification model** to recognize Bharatanatyam mudras (hand gestures):
- **Musti** | **Pataka** | **Sikhara** | **Simhamukha** | **Trisula**

**Dataset:** 2000 images (400 per class)  
**Approach:** Custom CNN + Transfer Learning comparison (MobileNetV2, EfficientNetB0, ResNet50)

---

## Project Workflow

1. **Data Analysis & EDA** - Class distribution, image characteristics analysis
2. **Preprocessing & Augmentation** - Image resizing, normalization, data augmentation pipeline
3. **Custom CNN Development** - Design, train, and evaluate custom CNN model
4. **Model Evaluation** - Accuracy, precision, recall, F1-score, confusion matrix, misclassified image analysis
5. **Transfer Learning (Bonus)** - Compare MobileNetV2, EfficientNetB0, ResNet50 with custom CNN
6. **Model Deployment** - Streamlit web application

---

## Repository Structure

```
Exit_Exam/
├── MUDRA_CNN_Project.ipynb          # Main Colab notebook (all analysis & training)
├── mudra_cnn_model.keras            # Trained custom CNN model
├── label_encoder.pkl                # Class label encoder
├── model_metadata.json              # Model metrics and configuration
├── app.py                           # Streamlit deployment application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── screenshots/                     # Results & visualizations
    ├── class_distribution.png
    ├── sample_images.png
    ├── augmentation_samples.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── misclassified_images.png
    └── streamlit_app_screenshots/
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Google Colab (for training)
- GPU recommended (for faster training)

### Local Setup (for Streamlit app)

```bash
# Clone repository
git clone https://github.com/SruthiGS-Gito/Exit_Exam.git
cd Exit_Exam

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training the Model

### Step 1: Run Colab Notebook
1. Open `MUDRA_CNN_Project.ipynb` in Google Colab
2. Execute cells sequentially:
   - **Cell 1-2:** Load libraries and dataset
   - **Cell 3-5:** Data analysis (class distribution, image characteristics)
   - **Cell 6-8:** Preprocessing and augmentation
   - **Cell 9-13:** Train custom CNN
   - **Cell 14-17:** Evaluate model and analyze failures
   - **Cell 18-19:** Save model and preprocessors
   - **Cell 20+:** Transfer learning comparison (bonus)

### Step 2: Download Artifacts
After training, download:
- `mudra_cnn_model.keras`
- `label_encoder.pkl`
- `model_metadata.json`

Place them in the project directory.

---

## Model Performance

### Custom CNN Results
| Metric | Value |
|--------|-------|
| Accuracy | ~85-90% |
| Precision | ~85-90% |
| Recall | ~85-90% |
| F1-Score | ~85-90% |

### Key Findings
- **Dataset:** Perfectly balanced (400 images per class)
- **Most confused pair:** (visual analysis from confusion matrix)
- **Misclassification pattern:** High-confidence mistakes → classes visually similar
- **Main challenge:** Hand angle variations and lighting conditions

---

## Deployment - Streamlit App

### Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Features
- **Home Page:** Dataset overview, model info, performance metrics
- **Prediction Page:** Upload mudra image for classification
- **Results Page:** Display predicted class, confidence score, top-3 predictions

---

## Analytical Questions Answered

### Data Analysis
**Q: Is the dataset balanced?**  
A: Yes, perfectly balanced - 400 images per class (20% each)

**Q: What image characteristics make mudra recognition difficult?**  
A: 
1. Hand angle and position - Same mudra looks different at different angles
2. Lighting and shadows - Poor lighting obscures finger details

### Model Development
**Q: Did the model show overfitting or underfitting?**  
A: Analyzed using learning curves - (see `training_curves.png`)

### Preprocessing
**Q: Which augmentations preserve semantic meaning?**  
A: Rotation, shift, zoom, brightness changes (preserves hand gesture)  
**Q: Which augmentations to avoid?**  
A: Horizontal/vertical flip - mirrors hand orientation (breaks semantics)

### Failure Analysis
**Q: Are misclassified samples high-confidence or low-confidence?**  
A: (See `misclassified_images.png` with confidence scores)

---

## Transfer Learning Comparison (Bonus)

Models compared:
- Custom CNN
- MobileNetV2
- EfficientNetB0
- ResNet50

**Metrics:** Accuracy, Model Size, Training Time, Inference Speed

**Recommended deployment model:** (justification based on trade-offs)

---

## Key Files

| File | Purpose |
|------|---------|
| `MUDRA_CNN_Project.ipynb` | Complete training & analysis pipeline |
| `mudra_cnn_model.keras` | Trained model (ready for inference) |
| `label_encoder.pkl` | Class encoding for predictions |
| `app.py` | Streamlit web application |
| `requirements.txt` | All dependencies |

---

## Requirements

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
Pillow>=8.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
streamlit>=1.0.0
```

---

## Author
**Sruthi GS**  
Exit Exam - ML/AI Assessment  
Date: September 2026

---

## Notes
- Training on full dataset takes ~30-45 minutes on Colab GPU
- Model input size: 224×224×3 (RGB images)
- Output: 5-class probability distribution (softmax)
- All visualizations saved in `screenshots/` folder

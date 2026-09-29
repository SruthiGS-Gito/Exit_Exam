import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import json
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Mudra Recognition",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_model_and_encoder():
    """Load trained model and label encoder"""
    try:
        model = tf.keras.models.load_model('mudra_cnn_model.keras')
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, encoder, metadata
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure mudra_cnn_model.keras, label_encoder.pkl, and model_metadata.json are in the app directory.")
        st.stop()

# Load model
model, label_encoder, metadata = load_model_and_encoder()

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Choose a page:", ["Home", "Predict", "About Model"])

# ==================== HOME PAGE ====================
if page == "Home":
    st.markdown("<div class='header'><h1>🙏 Bharatanatyam Mudra Recognition</h1></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("📊 Dataset Overview")
        st.write(f"**Total Images:** 2000")
        st.write(f"**Classes:** {len(metadata['classes'])}")
        st.write(f"**Images per class:** 400")
        st.write(f"**Train/Val/Test Split:** 70% / 15% / 15%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Mudra Classes")
        classes_text = " | ".join(metadata['classes'])
        st.write(f"**{classes_text}**")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Performance
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metadata['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metadata['recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{metadata['f1_score']:.2%}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # About Mudras
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🎭 About Mudras")
    st.write("""
    Mudras are symbolic hand gestures used in Bharatanatyam classical Indian dance. 
    Each mudra has specific meaning and conveys different emotions and stories.
    
    **Classes in this model:**
    - **Musti** - Fist gesture
    - **Pataka** - Flag gesture
    - **Sikhara** - Peak gesture
    - **Simhamukha** - Lion face gesture
    - **Trisula** - Trident gesture
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== PREDICTION PAGE ====================
elif page == "Predict":
    st.markdown("<h1>🔮 Predict Mudra</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card'>
    Upload an image of a hand gesture and the model will predict which mudra it is.
    </div>
    """, unsafe_allow_html=True)
    
    # Upload image
    uploaded_file = st.file_uploader("📤 Upload mudra image (JPG, PNG):", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        # Preprocess and predict
        with col2:
            st.subheader("🎯 Prediction Result")
            
            # Resize image to model input size
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction_proba = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(prediction_proba[0])
            predicted_class = label_encoder.classes_[predicted_class_idx]
            confidence = prediction_proba[0][predicted_class_idx]
            
            # Display prediction
            st.success(f"✅ Predicted Mudra: **{predicted_class}**")
            st.info(f"🎯 Confidence: **{confidence:.2%}**")
            
            # Top 3 predictions
            st.subheader("🏆 Top 3 Predictions")
            top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
            
            for rank, idx in enumerate(top_3_indices, 1):
                class_name = label_encoder.classes_[idx]
                prob = prediction_proba[0][idx]
                st.write(f"{rank}. **{class_name}** - {prob:.2%}")
            
            # Confidence bar chart
            st.subheader("📊 All Predictions")
            pred_dict = {label_encoder.classes_[i]: prediction_proba[0][i] for i in range(len(label_encoder.classes_))}
            st.bar_chart(pred_dict)

# ==================== ABOUT MODEL PAGE ====================
elif page == "About Model":
    st.markdown("<h1>ℹ️ About the Model</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🏗️ Model Architecture")
    st.write("""
    **Custom CNN** - Convolutional Neural Network
    
    **Architecture:**
    - 4 Convolutional blocks with BatchNormalization
    - Progressive filter increase: 32 → 64 → 128 → 256
    - Dropout layers to prevent overfitting
    - 2 Dense layers (512 → 256)
    - Output: 5-class Softmax
    
    **Input:** 224×224×3 RGB images  
    **Output:** Probability distribution across 5 mudra classes
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("📚 Training Details")
    st.write(f"""
    **Optimizer:** Adam (lr=0.001)  
    **Loss Function:** Sparse Categorical Crossentropy  
    **Batch Size:** 32  
    **Epochs:** 50 (with early stopping)  
    **Validation Split:** 15% of training data  
    **Test Set:** 15% of total data  
    
    **Callbacks:**
    - Early Stopping (patience=10)
    - Learning Rate Reduction (factor=0.5)
    - Model Checkpoint (best weights saved)
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("📊 Data Preprocessing")
    st.write("""
    **Image Resizing:** All images resized to 224×224 pixels  
    **Normalization:** Pixel values scaled to 0-1 range  
    **Augmentation (Training Only):**
    - Rotation: ±20°
    - Shift: ±15% (width & height)
    - Zoom: ±15%
    - Brightness: ±20%
    
    **Note:** No horizontal/vertical flips (preserves gesture semantics)
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("⚠️ Known Challenges")
    st.write("""
    1. **Hand angle variations** - Same mudra at different angles
    2. **Lighting conditions** - Poor lighting obscures finger details
    3. **Similar mudra pairs** - Some mudras share visual similarities
    4. **Occlusion** - Fingers hidden behind other body parts
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🚀 Model Info")
    st.write(f"""
    **Total Parameters:** ~7.2M  
    **Model Type:** Custom CNN  
    **Framework:** TensorFlow/Keras  
    **Input Shape:** (224, 224, 3)  
    **Number of Classes:** {len(metadata['classes'])}  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
    <p>Bharatanatyam Mudra Recognition</p>
    <p>Built with Streamlit & TensorFlow</p>
    <p>© 2026 - Sruthi GS</p>
    </div>
""", unsafe_allow_html=True)

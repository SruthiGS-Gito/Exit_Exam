import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
except ImportError as error:
    tf = None
    TENSORFLOW_IMPORT_ERROR = str(error)
else:
    TENSORFLOW_IMPORT_ERROR = ""


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "custom_cnn_mudra_model.keras"
CLASS_INDICES_PATH = APP_DIR / "class_indices.json"
METADATA_PATH = APP_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Mudra Recognition",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 2rem; }
    .stApp { background-color: #f0f2f6; }
    .header { text-align: center; color: #1f77b4; margin-bottom: 2rem; }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_app_data():
    classes = []
    class_error = ""
    metadata = {}
    metadata_error = ""

    try:
        with CLASS_INDICES_PATH.open(encoding="utf-8") as file:
            class_indices = json.load(file)
        ordered_indices = sorted(
            ((int(index), name) for name, index in class_indices.items()),
            key=lambda item: item[0],
        )
        if [index for index, _ in ordered_indices] != list(range(len(ordered_indices))):
            raise ValueError("Class indices must be unique and start at zero.")
        classes = [name for _, name in ordered_indices]
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as error:
        class_error = f"Could not load class order from {CLASS_INDICES_PATH.name}: {error}"

    try:
        with METADATA_PATH.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object.")
    except FileNotFoundError:
        metadata = {}
    except (OSError, json.JSONDecodeError, ValueError) as error:
        metadata_error = f"Could not read {METADATA_PATH.name}: {error}"

    model = None
    model_error = ""
    if not MODEL_PATH.is_file():
        model_error = (
            f"The trained model artifact is unavailable ({MODEL_PATH.name}). "
            "Prediction cannot currently be performed."
        )
    elif tf is None:
        model_error = f"TensorFlow is unavailable, so the model cannot be loaded: {TENSORFLOW_IMPORT_ERROR}"
    else:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as error:
            model_error = f"Could not load {MODEL_PATH.name}: {error}"

    return model, classes, metadata, class_error, metadata_error, model_error


model, classes, metadata, class_error, metadata_error, model_error = load_app_data()
input_width = int(metadata.get("IMG_WIDTH", 128))
input_height = int(metadata.get("IMG_HEIGHT", 128))

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Choose a page:", ["Home", "Predict", "About Model"])

if class_error:
    st.error(class_error)
if metadata_error:
    st.warning(metadata_error)
if model_error:
    st.warning(model_error)

if page == "Home":
    st.markdown(
        "<div class='header'><h1>🙏 Bharatanatyam Mudra Recognition</h1></div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("📊 Model Overview")
        st.write(f"**Input:** {input_width}×{input_height} RGB images")
        st.write("**Preprocessing:** pixel values scaled to 0–1")
        st.write(f"**Model artifact:** {MODEL_PATH.name}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Mudra Classes")
        st.write(" | ".join(classes) if classes else "Class labels are unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🎭 About Mudras")
    st.write(
        "Mudras are symbolic hand gestures used in Bharatanatyam classical Indian dance. "
        "This application is configured for the classes listed above."
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Predict":
    st.markdown("<h1>🔮 Predict Mudra</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='metric-card'>
        Upload an image of a hand gesture to see the model prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "📤 Upload mudra image (JPG, PNG):", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🎯 Prediction Result")
            if model is None:
                st.info("Prediction is disabled until the trained model artifact is available.")
            elif not classes:
                st.error("Prediction is disabled because the class order could not be loaded.")
            else:
                resized_image = image.resize(
                    (input_width, input_height), resample=Image.Resampling.NEAREST
                )
                image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                probabilities = model.predict(image_array, verbose=0)[0]

                if len(probabilities) != len(classes):
                    st.error(
                        "The model output count does not match the class indices file."
                    )
                else:
                    predicted_index = int(np.argmax(probabilities))
                    st.success(f"✅ Predicted Mudra: **{classes[predicted_index]}**")
                    st.info(f"🎯 Confidence: **{probabilities[predicted_index]:.2%}**")

                    st.subheader("🏆 Top Predictions")
                    top_indices = np.argsort(probabilities)[-min(3, len(classes)):][::-1]
                    for rank, index in enumerate(top_indices, 1):
                        st.write(f"{rank}. **{classes[index]}** - {probabilities[index]:.2%}")
                    st.subheader("📊 All Predictions")
                    st.bar_chart(
                        {classes[index]: float(probabilities[index]) for index in range(len(classes))}
                    )

elif page == "About Model":
    st.markdown("<h1>ℹ️ About the Model</h1>", unsafe_allow_html=True)

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🏗️ Model Architecture")
    st.write(
        """
        **Custom CNN**

        - Conv2D blocks with 32, 64, and 128 filters, each followed by max pooling
        - Flatten, Dense(128, ReLU), Dropout(0.5)
        - Five-class softmax output

        **Input:** 128×128×3 RGB images
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("📚 Training and Preprocessing")
    st.write(
        """
        **Optimizer:** Adam (learning rate 0.001)  
        **Loss:** Categorical crossentropy  
        **Batch size:** 32  
        **Training:** Up to 10 epochs, with early stopping and learning-rate reduction  
        **Inference preprocessing:** nearest-neighbor resize to 128×128, RGB, pixel values scaled to 0–1

        Training augmentation included rotation, shifts, shear, zoom, and horizontal flipping.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🚀 Model Status")
    st.write(f"**Classes:** {', '.join(classes) if classes else 'Unavailable'}")
    st.write(f"**Model file:** {MODEL_PATH.name}")
    if model_error:
        st.warning(model_error)
    else:
        st.write("The trained model is loaded and ready for prediction.")
    st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
    <p>Bharatanatyam Mudra Recognition</p>
    <p>Built with Streamlit & TensorFlow</p>
    <p>© 2026 - Sruthi GS</p>
    </div>
    """,
    unsafe_allow_html=True,
)

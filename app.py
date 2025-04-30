import streamlit as st
import tensorflow as tf
import gdown
import os
import cv2
import numpy as np
from PIL import Image
import magic
from datasets import load_dataset
import tempfile

# Define model options and corresponding Google Drive File IDs
MODEL_OPTIONS = {
    "CNN": ("1---NhvKS9H-c5yf04hB8NzOrtIpFcKL8", "keras"),  # Replace with actual File ID
    "ResNet50": ("1nv2I-K8QKbGc62eQDx5OLcRYinjJPXai", "keras")
}

# Fetch id2label mapping from Hugging Face dataset
@st.cache_data
def get_class_labels():
    try:
        dataset = load_dataset("kannanwisen/Indian-Traffic-Sign-Classification")
        class_names = dataset['train'].features['label'].names
        id2label = {i: label for i, label in enumerate(class_names)}
        return id2label
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return {}

# Streamlit UI
st.title("Traffic Sign Classification")
st.write("Select a model and upload an image for classification.")

# Select model
selected_model_name = st.selectbox("Select a model:", list(MODEL_OPTIONS.keys()))

if selected_model_name:
    selected_model_id, selected_model_type = MODEL_OPTIONS[selected_model_name]
else:
    st.error("Please select a model!")
    st.stop()

# Function to load model with caching
@st.cache_resource(hash_funcs={str: lambda _: None})
def load_model(model_id, model_type, _cache_buster=str(os.urandom(16))):
    url = f"https://drive.google.com/uc?id={model_id}"
    output = f"{model_id}.keras"
    
    with st.spinner(f"Downloading model from {url}..."):
        try:
            gdown.download(url, output, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"Download failed: {e}. Please check the Google Drive link or network connection.")
            return None
    
    if not os.path.exists(output):
        st.error(f"File {output} not found after download!")
        return None
    
    st.write(f"File size: {os.path.getsize(output)} bytes")
    try:
        file_type = magic.from_file(output)
        st.write(f"Detected file type: {file_type}")
        if "keras" not in file_type.lower():
            st.error(f"Expected a Keras model file, but got: {file_type}")
            return None
    except Exception as e:
        st.error(f"Failed to detect file type: {e}")
        return None
    
    try:
        model = tf.keras.models.load_model(output)
        st.write("Keras model (.keras format) loaded successfully.")
    except ValueError as ve:
        st.error(f"Failed to load Keras model. Possible version mismatch or corrupted file: {ve}")
        return None
    except Exception as e:
        st.error(f"Failed to load Keras model: {e}")
        return None
    
    return model

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None

# Load model button
if st.button("Load Model"):
    with st.spinner("Loading model..."):
        st.session_state.model = load_model(selected_model_id, selected_model_type)
        if st.session_state.model:
            st.success(f"✅ {selected_model_name} Loaded Successfully!")
        else:
            st.error("❌ Model loading failed.")

id2label = get_class_labels()
if not id2label:
    st.error("❌ Failed to load class labels. Cannot proceed with classification.")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Preprocess image function
def preprocess_test_image(image_path, model_name):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if model_name == "ResNet50":
        image = cv2.resize(image, (224, 224))  # ResNet50 input size
        image = tf.keras.applications.resnet50.preprocess_input(image)
    else:
        image = cv2.resize(image, (64, 64))  # Default for CNN
        image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# Display and classify image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        if st.session_state.model is None:
            st.error("❌ Please load a model first!")
        else:
            try:
                # Convert PIL image to OpenCV format
                image_cv = np.array(image)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, image_cv)
                    input_data = preprocess_test_image(temp_path, selected_model_name)
                
                # Classify
                with st.spinner("Classifying image..."):
                    prediction = st.session_state.model.predict(input_data)
                    predicted_class_idx = int(np.argmax(prediction))
                    predicted_label = id2label.get(predicted_class_idx, "Unknown")
                    st.success(f"Prediction: {predicted_label} (Class ID: {predicted_class_idx})")
            except Exception as e:
                st.error(f"Classification failed: {e}")

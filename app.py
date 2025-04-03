import streamlit as st
import tensorflow as tf
import gdown
import torch
import os
import cv2
import numpy as np
from PIL import Image
import magic

# Define model options and corresponding Google Drive File IDs
MODEL_OPTIONS = {
    "CNN": ("1---NhvKS9H-c5yf04hB8NzOrtIpFcKL8", "keras"),  # Replace with actual File ID of model.keras
    "ResNet50": ("1nv2I-K8QKbGc62eQDx5OLcRYinjJPXai", "keras")
}

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
@st.cache_resource
def load_model(model_id, model_type):
    url = f"https://drive.google.com/uc?id={model_id}"
    output = "model.keras"  # Download directly as model.keras
    
    st.write(f"Downloading model from {url} to {output}...")
    try:
        gdown.download(url, output, quiet=False, fuzzy=True)
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None
    
    if not os.path.exists(output):
        st.error(f"File {output} not found after download!")
        return None
    
    st.write(f"File size: {os.path.getsize(output)} bytes")
    file_type = magic.from_file(output)
    st.write(f"Detected file type: {file_type}")
    
    if model_type == "keras":
        try:
            model = tf.keras.models.load_model(output)
            st.write("Keras model (.keras format) loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load Keras model: {e}")
            return None
    else:
        try:
            model = torch.load(output, map_location=torch.device('cpu'))
            model.eval()
            st.write("PyTorch model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load PyTorch model: {e}")
            return None
    
    return model

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None

# Load model button
if st.button("Load Model"):
    st.session_state.model = load_model(selected_model_id, selected_model_type)
    if st.session_state.model:
        st.success(f"✅ {selected_model_name} Loaded Successfully!")
    else:
        st.error("❌ Model loading failed.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Preprocess image function
def preprocess_test_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))  # Adjust to model input size
    image = image.astype('float32') / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Display and classify image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        if st.session_state.model:
            try:
                # Convert PIL image to OpenCV format
                image_cv = np.array(image)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                
                # Save to a temporary file to use preprocess_test_image
                temp_path = "temp_image.jpg"
                cv2.imwrite(temp_path, image_cv)
                
                # Preprocess image
                input_data = preprocess_test_image(temp_path)
                
                if selected_model_type == "keras":
                    prediction = st.session_state.model.predict(input_data)
                else:
                    with torch.no_grad():
                        prediction = st.session_state.model(torch.tensor(input_data, dtype=torch.float32)).numpy()
                
                st.write("Prediction:", np.argmax(prediction))  # Adjust based on your class labels
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.error("❌ Please load a model first!")

import streamlit as st
import tensorflow as tf
import gdown
import torch
import os
import h5py
from PIL import Image
import numpy as np
import magic  # For file type detection

# Define model options and corresponding Google Drive File IDs
MODEL_OPTIONS = {
    "CNN": ("1---NhvKS9H-c5yf04hB8NzOrtIpFcKL8", "keras"),  # Replace with actual File ID
    "ResNet50": ("1nv2I-K8QKbGc62eQDx5OLcRYinjJPXai", "keras")  # Replace with actual File ID
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
    output = f"model.{ 'h5' if model_type == 'keras' else 'pth' }"
    st.write(f"Downloading model from {url} to {output}...")
    
    try:
        gdown.download(url, output, quiet=False, fuzzy=True)
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None
    
    if not os.path.exists(output):
        st.error(f"File {output} not found after download!")
        return None
    
    file_size = os.path.getsize(output)
    st.write(f"File size: {file_size} bytes")
    
    if model_type == "keras":
        try:
            with h5py.File(output, 'r') as f:
                st.write("Verified as HDF5 file.")
            model = tf.keras.models.load_model(output)
            st.write("Keras model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load Keras model: {e}")
            try:
                file_type = magic.from_file(output)
                st.write(f"Detected file type: {file_type}")
            except:
                st.write("Could not determine file type.")
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
def preprocess_image(image, model_type):
    image = image.resize((224, 224))  # Adjust size based on your model's input requirements
    image_array = np.array(image) / 255.0  # Normalize
    
    if model_type == "keras":
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension for Keras
    else:  # PyTorch
        image_array = np.transpose(image_array, (2, 0, 1))  # Change from HWC to CHW
        image_array = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
    
    return image_array

# Display and classify image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        if st.session_state.model:
            try:
                input_data = preprocess_image(image, selected_model_type)
                
                if selected_model_type == "keras":
                    prediction = st.session_state.model.predict(input_data)
                else:
                    with torch.no_grad():
                        prediction = st.session_state.model(input_data).numpy()
                
                st.write("Prediction:", np.argmax(prediction))  # Adjust based on your class labels
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.error("❌ Please load a model first!")

import streamlit as st
import tensorflow as tf
import gdown
import torch
import os
from PIL import Image
import numpy as np
import magic
import cv2  # Ensure using opencv-python-headless on Streamlit Cloud

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
def preprocess_image(image, model_type):
    image = image.resize((64, 64))  # Resize based on your model's expected input size
    image_array = np.array(image) / 255.0  # Normalize
    
    if model_type == "keras":
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension for Keras
    else:  # PyTorch
        image_array = np.transpose(image_array, (2, 0, 1))  # Change from HWC to CHW
        image_array = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
    
    return image_array

# Class label mapping
class_names = {
    'ALL_MOTOR_VEHICLE_PROHIBITED': 0, 'AXLE_LOAD_LIMIT': 1, 'BARRIER_AHEAD': 2, 'BULLOCK_AND_HANDCART_PROHIBITED': 3, 'BULLOCK_PROHIBITED': 4,
    'CATTLE': 5, 'COMPULSARY_AHEAD': 6, 'COMPULSARY_AHEAD_OR_TURN_LEFT': 7, 'COMPULSARY_AHEAD_OR_TURN_RIGHT': 8, 'COMPULSARY_CYCLE_TRACK': 9,
    'COMPULSARY_KEEP_LEFT': 10, 'COMPULSARY_KEEP_RIGHT': 11, 'COMPULSARY_MINIMUM_SPEED': 12, 'COMPULSARY_SOUND_HORN': 13, 'COMPULSARY_TURN_LEFT': 14,
    'COMPULSARY_TURN_LEFT_AHEAD': 15, 'COMPULSARY_TURN_RIGHT': 16, 'COMPULSARY_TURN_RIGHT_AHEAD': 17, 'CROSS_ROAD': 18, 'CYCLE_CROSSING': 19,
    'CYCLE_PROHIBITED': 20, 'DANGEROUS_DIP': 21, 'DIRECTION': 22, 'FALLING_ROCKS': 23, 'FERRY': 24, 'GAP_IN_MEDIAN': 25, 'GIVE_WAY': 26,
    'GUARDED_LEVEL_CROSSING': 27, 'HANDCART_PROHIBITED': 28, 'HEIGHT_LIMIT': 29, 'HORN_PROHIBITED': 30, 'HUMP_OR_ROUGH_ROAD': 31, 'LEFT_HAIR_PIN_BEND': 32,
    'LEFT_HAND_CURVE': 33, 'LEFT_REVERSE_BEND': 34, 'LEFT_TURN_PROHIBITED': 35, 'LENGTH_LIMIT': 36, 'LOAD_LIMIT': 37, 'LOOSE_GRAVEL': 38, 'MEN_AT_WORK': 39,
    'NARROW_BRIDGE': 40, 'NARROW_ROAD_AHEAD': 41, 'NO_ENTRY': 42, 'NO_PARKING': 43, 'NO_STOPPING_OR_STANDING': 44, 'OVERTAKING_PROHIBITED': 45,
    'PASS_EITHER_SIDE': 46, 'PEDESTRIAN_CROSSING': 47, 'PEDESTRIAN_PROHIBITED': 48, 'PRIORITY_FOR_ONCOMING_VEHICLES': 49, 'QUAY_SIDE_OR_RIVER_BANK': 50,
    'RESTRICTION_ENDS': 51, 'RIGHT_HAIR_PIN_BEND': 52, 'RIGHT_HAND_CURVE': 53, 'RIGHT_REVERSE_BEND': 54, 'RIGHT_TURN_PROHIBITED': 55, 'ROAD_WIDENS_AHEAD': 56,
    'ROUNDABOUT': 57, 'SCHOOL_AHEAD': 58, 'SIDE_ROAD_LEFT': 59, 'SIDE_ROAD_RIGHT': 60, 'SLIPPERY_ROAD': 61, 'SPEED_LIMIT_15': 62, 'SPEED_LIMIT_20': 63,
    'SPEED_LIMIT_30': 64, 'SPEED_LIMIT_40': 65, 'SPEED_LIMIT_5': 66, 'SPEED_LIMIT_50': 67, 'SPEED_LIMIT_60': 68, 'SPEED_LIMIT_70': 69, 'SPEED_LIMIT_80': 70,
    'STAGGERED_INTERSECTION': 71, 'STEEP_ASCENT': 72, 'STEEP_DESCENT': 73, 'STOP': 74, 'STRAIGHT_PROHIBITED': 75, 'TONGA_PROHIBITED': 76, 'TRAFFIC_SIGNAL': 77,
    'TRUCK_PROHIBITED': 78, 'TURN_RIGHT': 79, 'T_INTERSECTION': 80, 'UNGUARDED_LEVEL_CROSSING': 81, 'U_TURN_PROHIBITED': 82, 'WIDTH_LIMIT': 83, 'Y_INTERSECTION': 84
}

# Display and classify image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        if st.session_state.model:
            try:
                input_data = preprocess_image(image, selected_model_type)
                
                # Inference
                if selected_model_type == "keras":
                    prediction = st.session_state.model.predict(input_data)
                else:
                    with torch.no_grad():
                        prediction = st.session_state.model(input_data).numpy()
                
                # Map prediction to class label
                predicted_label_index = np.argmax(prediction)
                predicted_label = list(class_names.keys())[predicted_label_index]
                
                st.write("Prediction: ", predicted_label)
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.error("❌ Please load a model first!")

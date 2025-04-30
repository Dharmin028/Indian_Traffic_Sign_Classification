import streamlit as st
import tensorflow as tf
import gdown
import torch
import os
import cv2
import numpy as np
from PIL import Image
import magic
# Class ID to Label mapping
id2label = {
    0: 'ALL_MOTOR_VEHICLE_PROHIBITED', 1: 'AXLE_LOAD_LIMIT', 2: 'BARRIER_AHEAD',
    3: 'BULLOCK_AND_HANDCART_PROHIBITED', 4: 'BULLOCK_PROHIBITED', 5: 'CATTLE',
    6: 'CATTLE_PROHIBITED', 7: 'CHECK_POST', 8: 'COMPULSORY_AHEAD', 9: 'COMPULSORY_AHEAD_OR_TURN_LEFT',
    10: 'COMPULSORY_KEEP_LEFT', 11: 'COMPULSORY_KEEP_RIGHT', 12: 'COMPULSORY_LEFT_TURN',
    13: 'COMPULSORY_RIGHT_TURN', 14: 'CYCLE_CROSSING', 15: 'CYCLE_PROHIBITED', 16: 'DANGEROUS_DIP',
    17: 'DEAD_END', 18: 'DEER_CROSSING', 19: 'DIRECTION_SIGN', 20: 'DIVERSION', 21: 'FALLING_ROCKS',
    22: 'FERRY', 23: 'FERRY_PROHIBITED', 24: 'FOOTPATH', 25: 'GAP_IN_MEDIAN', 26: 'GIVEWAY',
    27: 'GUARD_POST', 28: 'HAIRPIN_BEND_LEFT', 29: 'HAIRPIN_BEND_RIGHT', 30: 'HANDCART_PROHIBITED',
    31: 'HORN_PROHIBITED', 32: 'HUMP_OR_ROUGH', 33: 'KEEP_LEFT', 34: 'LEFT_HAIR_PIN_BEND',
    35: 'LEFT_REVERSE_BEND', 36: 'LEFT_TURN_PROHIBITED', 37: 'LEFTHAND_CURVE', 38: 'LOAD_LIMIT',
    39: 'LOOSE_GRAVEL', 40: 'MEN_AT_WORK', 41: 'NARROW_BRIDGE', 42: 'NARROW_ROAD_AHEAD',
    43: 'NO_ENTRY', 44: 'NO_PARKING', 45: 'NO_STOPPING_NO_STANDING', 46: 'ONE_WAY_SIGN',
    47: 'OVERTAKING_PROHIBITED', 48: 'PARKING', 49: 'PEDESTRAIN_CROSSING', 50: 'PEDESTRAINS_PROHIBITED',
    51: 'PETROL_PUMP', 52: 'RESTRICTION_ENDS', 53: 'RIGHT_HAIR_PIN_BEND', 54: 'RIGHT_REVERSE_BEND',
    55: 'RIGHT_TURN_PROHIBITED', 56: 'RIGHTHAND_CURVE', 57: 'ROAD_WIDENS', 58: 'ROUNDABOUT',
    59: 'SCHOOL_AHEAD', 60: 'SIDE_ROAD_LEFT', 61: 'SIDE_ROAD_RIGHT', 62: 'SLIPPERY_ROAD',
    63: 'SPEED_LIMIT_20', 64: 'SPEED_LIMIT_30', 65: 'SPEED_LIMIT_40', 66: 'SPEED_LIMIT_50',
    67: 'SPEED_LIMIT_60', 68: 'SPEED_LIMIT_70', 69: 'SPEED_LIMIT_80', 70: 'SPEED_LIMIT_90',
    71: 'SPEED_LIMIT_REMOVER', 72: 'STOP', 73: 'STRAIGHT_PROHIBITED', 74: 'STRAIGHTPROHIBITED_LEFT_TURN',
    75: 'STRAIGHTPROHIBITED_RIGHT_TURN', 76: 'T_INTERSECTION', 77: 'TRAFFIC_SIGNAL_AHEAD',
    78: 'TRUCK_PROHIBITED', 79: 'TURN_LEFT', 80: 'TURN_RIGHT', 81: 'U_TURN_PROHIBITED',
    82: 'UNGUARDED_LEVEL_CROSSING', 83: 'WIDTH_LIMIT', 84: 'Y_INTERSECTION'
}

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
                
                predicted_class = int(np.argmax(prediction))
                label = id2label.get(predicted_class, "Unknown")
                st.success(f"Prediction: {label} (Class ID: {predicted_class})")
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.error("❌ Please load a model first!")

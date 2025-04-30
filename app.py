import streamlit as st
import tensorflow as tf
import gdown
import torch
import os
from PIL import Image
import numpy as np
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

# Preprocess image function using PIL
def preprocess_image(image, model_type):
    # Convert image to RGB (in case it's not already)
    image = image.convert('RGB')
    # Resize image to the model's expected input size (224x224)
    image = image.resize((224, 224))
    # Normalize image values to [0, 1] range
    image_array = np.array(image) / 255.0
    
    # Adjust image dimensions depending on model type
    if model_type == "keras":
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension for Keras
    else:  # For PyTorch
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
                
                # Map the index to class label
                class_names = {
                    0: 'ALL_MOTOR_VEHICLE_PROHIBITED', 1: 'AXLE_LOAD_LIMIT', 2: 'BARRIER_AHEAD', 3: 'BULLOCK_AND_HANDCART_PROHIBITED',
                    4: 'BULLOCK_PROHIBITED', 5: 'CATTLE', 6: 'COMPULSARY_AHEAD', 7: 'COMPULSARY_AHEAD_OR_TURN_LEFT',
                    8: 'COMPULSARY_AHEAD_OR_TURN_RIGHT', 9: 'COMPULSARY_CYCLE_TRACK', 10: 'COMPULSARY_KEEP_LEFT',
                    11: 'COMPULSARY_KEEP_RIGHT', 12: 'COMPULSARY_MINIMUM_SPEED', 13: 'COMPULSARY_SOUND_HORN', 14: 'COMPULSARY_TURN_LEFT',
                    15: 'COMPULSARY_TURN_LEFT_AHEAD', 16: 'COMPULSARY_TURN_RIGHT', 17: 'COMPULSARY_TURN_RIGHT_AHEAD', 18: 'CROSS_ROAD',
                    19: 'CYCLE_CROSSING', 20: 'CYCLE_PROHIBITED', 21: 'DANGEROUS_DIP', 22: 'DIRECTION', 23: 'FALLING_ROCKS', 24: 'FERRY',
                    25: 'GAP_IN_MEDIAN', 26: 'GIVE_WAY', 27: 'GUARDED_LEVEL_CROSSING', 28: 'HANDCART_PROHIBITED', 29: 'HEIGHT_LIMIT',
                    30: 'HORN_PROHIBITED', 31: 'HUMP_OR_ROUGH_ROAD', 32: 'LEFT_HAIR_PIN_BEND', 33: 'LEFT_HAND_CURVE', 34: 'LEFT_REVERSE_BEND',
                    35: 'LEFT_TURN_PROHIBITED', 36: 'LENGTH_LIMIT', 37: 'LOAD_LIMIT', 38: 'LOOSE_GRAVEL', 39: 'MEN_AT_WORK', 40: 'NARROW_BRIDGE',
                    41: 'NARROW_ROAD_AHEAD', 42: 'NO_ENTRY', 43: 'NO_PARKING', 44: 'NO_STOPPING_OR_STANDING', 45: 'OVERTAKING_PROHIBITED',
                    46: 'PASS_EITHER_SIDE', 47: 'PEDESTRIAN_CROSSING', 48: 'PEDESTRIAN_PROHIBITED', 49: 'PRIORITY_FOR_ONCOMING_VEHICLES',
                    50: 'QUAY_SIDE_OR_RIVER_BANK', 51: 'RESTRICTION_ENDS', 52: 'RIGHT_HAIR_PIN_BEND', 53: 'RIGHT_HAND_CURVE', 54: 'RIGHT_REVERSE_BEND',
                    55: 'RIGHT_TURN_PROHIBITED', 56: 'ROAD_WIDENS_AHEAD', 57: 'ROUNDABOUT', 58: 'SCHOOL_AHEAD', 59: 'SIDE_ROAD_LEFT',
                    60: 'SIDE_ROAD_RIGHT', 61: 'SLIPPERY_ROAD', 62: 'SPEED_LIMIT_15', 63: 'SPEED_LIMIT_20', 64: 'SPEED_LIMIT_30',
                    65: 'SPEED_LIMIT_40', 66: 'SPEED_LIMIT_5', 67: 'SPEED_LIMIT_50', 68: 'SPEED_LIMIT_60', 69: 'SPEED_LIMIT_70',
                    70: 'SPEED_LIMIT_80', 71: 'STAGGERED_INTERSECTION', 72: 'STEEP_ASCENT', 73: 'STEEP_DESCENT', 74: 'STOP', 75: 'STRAIGHT_PROHIBITED',
                    76: 'TONGA_PROHIBITED', 77: 'TRAFFIC_SIGNAL', 78: 'TRUCK_PROHIBITED', 79: 'TURN_RIGHT', 80: 'T_INTERSECTION',
                    81: 'UNGUARDED_LEVEL_CROSSING', 82: 'U_TURN_PROHIBITED', 83: 'WIDTH_LIMIT', 84: 'Y_INTERSECTION'
                }

                predicted_label_index = np.argmax(prediction)
                predicted_label = class_names[predicted_label_index]  # Map index to label name
                st.write("Prediction:", predicted_label)
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.error("❌ Please load a model first!")

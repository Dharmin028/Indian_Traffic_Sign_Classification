import streamlit as st
import tensorflow as tf
import gdown
import os
import cv2
import numpy as np
from PIL import Image
import magic
import tempfile

# Define model options and corresponding Google Drive File IDs
MODEL_OPTIONS = {
    "CNN": ("1---NhvKS9H-c5yf04hB8NzOrtIpFcKL8", "keras"),  # Replace with actual File ID
    "ResNet50": ("1nv2I-K8QKbGc62eQDx5OLcRYinjJPXai", "keras")
}

# Hardcoded id2label mapping (provided by user)
id2label = {
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
    output = f"{selected_model_name.lower()}_model.keras"  # Unique filename per model
    
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
    try:
        file_type = magic.from_file(output)
        st.write(f"Detected file type: {file_type}")
    except Exception as e:
        st.error(f"Failed to detect file type: {e}")
        return None
    
    # Handle case where file is detected as ZIP
    if "zip" in file_type.lower():
        st.warning("Downloaded file is a ZIP archive. Attempting to extract .keras file...")
        extract_dir = f"extracted_{selected_model_name.lower()}"
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            st.write(f"Extracted ZIP to {extract_dir}")
        except Exception as e:
            st.error(f"Failed to extract ZIP file: {e}")
            return None
        
        # Find .keras file
        keras_files = glob.glob(os.path.join(extract_dir, "*.keras"))
        if not keras_files:
            st.error("No .keras file found in the ZIP archive!")
            return None
        keras_file = keras_files[0]
    else:
        keras_file = output
    
    # Validate and load Keras model
    try:
        file_type = magic.from_file(keras_file)
        st.write(f"Detected file type for {keras_file}: {file_type}")
        if "keras" not in file_type.lower() and "zip" in file_type.lower():
            st.error(f"Expected a Keras model file, but got: {file_type}")
            return None
    except Exception as e:
        st.error(f"Failed to detect file type for {keras_file}: {e}")
        return None
    
    if model_type == "keras":
        try:
            model = tf.keras.models.load_model(keras_file)
            st.write("Keras model (.keras format) loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load Keras model: {e}")
            return None
    else:
        st.error("Only Keras models are supported in this configuration!")
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
def preprocess_test_image(image_path, model_name):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if model_name == "ResNet50":
        image = cv2.resize(image, (224, 224))  # ResNet50 input size
        image = tf.keras.applications.resnet50.preprocess_input(image)  # ResNet50 preprocessing
    else:
        image = cv2.resize(image, (64, 64))  # Default for CNN
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
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, image_cv)
                    input_data = preprocess_test_image(temp_path, selected_model_name)
                
                # Classify
                if selected_model_type == "keras":
                    prediction = st.session_state.model.predict(input_data)
                else:
                    st.error("Only Keras models are supported!")
                    st.stop()
                
                predicted_class_idx = int(np.argmax(prediction))
                predicted_label = id2label.get(predicted_class_idx, "Unknown")
                st.success(f"Prediction: {predicted_label} (Class ID: {predicted_class_idx})")
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.error("❌ Please load a model first!")

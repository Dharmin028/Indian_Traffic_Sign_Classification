import streamlit as st
import tensorflow as tf
import gdown
import torch
from PIL import Image
import numpy as np

MODEL_OPTIONS = {
    "CNN": ("1---NhvKS9H-c5yf04hB8NzOrtIpFcKL8", "keras"),
    "ResNet50": ("1nv2I-K8QKbGc62eQDx5OLcRYinjJPXai", "keras")
}

st.title("Traffic Sign Classification")
st.write("Select a model and upload an image for classification.")

selected_model_name = st.selectbox("Select a model:", list(MODEL_OPTIONS.keys()))

if selected_model_name:
    selected_model_id, selected_model_type = MODEL_OPTIONS[selected_model_name]
else:
    st.error("Please select a model!")
    st.stop()

@st.cache_resource
def load_model(model_id, model_type):
    url = f"https://drive.google.com/uc?id={model_id}"
    output = f"model.{ 'h5' if model_type == 'keras' else 'pth' }"
    try:
        gdown.download(url, output, quiet=False, fuzzy=True)
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return None
    
    try:
        if model_type == "keras":
            model = tf.keras.models.load_model(output)
        else:
            model = torch.load(output, map_location=torch.device('cpu'))
            model.eval()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

if 'model' not in st.session_state:
    st.session_state.model = None

if st.button("Load Model"):
    st.session_state.model = load_model(selected_model_id, selected_model_type)
    if st.session_state.model:
        st.success(f"✅ {selected_model_name} Loaded Successfully!")
    else:
        st.error("❌ Model loading failed.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def preprocess_image(image, model_type):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    if model_type == "keras":
        image_array = np.expand_dims(image_array, axis=0)
    else:
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
    return image_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        if st.session_state.model:
            input_data = preprocess_image(image, selected_model_type)
            if selected_model_type == "keras":
                prediction = st.session_state.model.predict(input_data)
            else:
                with torch.no_grad():
                    prediction = st.session_state.model(input_data).numpy()
            st.write("Prediction:", np.argmax(prediction))
        else:
            st.error("❌ Please load a model first!")

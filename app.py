import streamlit as st
import tensorflow as tf
import gdown
import torch

# Define model options and corresponding Google Drive File IDs
MODEL_OPTIONS = {
    "CNN": ("1---NhvKS9H-c5yf04hB8NzOrtIpFcKL8","keras"),  # Replace with actual File ID
    "ResNet50": ("1nv2I-K8QKbGc62eQDx5OLcRYinjJPXai","keras")  # Replace with another File ID
    # "VGG16" : ("1nv2I-K8QKbGc62eQDx5OLcRYinjJPXai","pytorch")
}

# Select model
selected_model, model_type = st.selectbox(
    "Select a model:", 
    list(MODEL_OPTIONS), 
    format_func=lambda x: x[0]
)

@st.cache_resource
def load_model(model_id, model_type):
    url = f"https://drive.google.com/uc?id={model_id}"
    output = f"model.{ 'h5' if model_type == 'keras' else 'pth' }"

    # Download model
    gdown.download(url, output, quiet=False)

    # Load Keras Model
    if model_type == "keras":
        model = tf.keras.models.load_model(output)
    # Load PyTorch Model
    else:
        model = torch.load(output, map_location=torch.device('cpu'))
        model.eval()

    return model

if st.button("Load Model"):
    model = load_model(selected_model[0], selected_model[1])
    st.write(f"✅ {selected_model[0]} Loaded Successfully!")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def preprocess_image(image, model_type):
    image = image.resize((64, 64))  # Resize for the model
    image_array = np.array(image) / 255.0  # Normalize
    
    if model_type == "keras":
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    else:  # PyTorch needs a different format
        image_array = np.transpose(image_array, (2, 0, 1))  # Change from HWC to CHW
        image_array = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
    
    return image_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        if 'model' in locals():
            input_data = preprocess_image(image, selected_model[1])
            
            if selected_model[1] == "keras":
                prediction = model.predict(input_data)
            else:
                with torch.no_grad():
                    prediction = model(input_data).numpy()
            
            st.write("Prediction:", np.argmax(prediction))
        else:
            st.error("❌ Please load a model first!")

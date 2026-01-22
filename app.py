import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------
# Load models
# -----------------------
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("custom_cnn_plant_disease.h5")
    mobilenet = tf.keras.models.load_model("mobilenetv2_plant_disease.h5")
    return cnn, mobilenet

cnn_model, mobilenet_model = load_models()

# -----------------------
# Class labels (PlantVillage)
# -----------------------
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___Healthy',
    'Blueberry___Healthy', 'Cherry___Healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Healthy',
    'Grape___Black_rot', 'Grape___Healthy', 'Peach___Healthy',
    'Pepper___Bacterial_spot', 'Pepper___Healthy',
    'Potato___Early_blight', 'Potato___Healthy', 'Potato___Late_blight',
    'Raspberry___Healthy', 'Soybean___Healthy',
    'Squash___Powdery_mildew', 'Strawberry___Healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌱 Plant Disease Detection System")
st.write("Upload a plant leaf image and select a model")

model_choice = st.selectbox(
    "Select Model",
    ["Custom CNN (From Scratch)", "MobileNetV2 (Transfer Learning)"]
)

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

# -----------------------
# Prediction function
# -----------------------
def predict(image, model, img_size):
    img = image.resize((img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = preds[0][idx] * 100
    return idx, confidence

# -----------------------
# Run prediction
# -----------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        if model_choice.startswith("Custom"):
            idx, conf = predict(image, cnn_model, 128)
            model_used = "Custom CNN"
        else:
            idx, conf = predict(image, mobilenet_model, 224)
            model_used = "MobileNetV2"

        st.success(f"🧠 Model Used: {model_used}")
        st.success(f"🌿 Predicted Disease: {CLASS_NAMES[idx]}")
        st.info(f"📊 Confidence: {conf:.2f}%")

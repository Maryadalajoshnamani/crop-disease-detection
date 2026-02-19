import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import streamlit as st

# Cache model (loads only once)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/plant_disease_model.h5")

model = load_model()

class_names = [
    "Pepper_Bacterial_spot",
    "Pepper_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Mosaic_Virus",
    "Tomato_healthy"
]

solutions = {
    "Pepper_Bacterial_spot": "Use copper based fungicide.",
    "Pepper_healthy": "Plant is healthy.",
    "Potato_Early_blight": "Spray Mancozeb fungicide.",
    "Potato_healthy": "Plant is healthy.",
    "Potato_Late_blight": "Apply Chlorothalonil.",
    "Tomato_Bacterial_spot": "Use copper spray.",
    "Tomato_Early_blight": "Spray Mancozeb weekly.",
    "Tomato_Late_blight": "Apply Metalaxyl.",
    "Tomato_Leaf_Mold": "Use sulfur spray.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves.",
    "Tomato_Spider_mites": "Use neem oil.",
    "Tomato_Target_Spot": "Apply Chlorothalonil.",
    "Tomato_YellowLeaf_Curl_Virus": "Control whiteflies.",
    "Tomato_Mosaic_Virus": "Remove infected plant.",
    "Tomato_healthy": "Plant is healthy."
}

def predict_disease(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    disease = class_names[class_index]
    solution = solutions[disease]

    return disease, solution

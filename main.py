import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# -----------------------------
# Base directory (DEPLOYMENT SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# CACHED LOADING (IMPORTANT)
# -----------------------------
@st.cache_resource
def load_embeddings():
    with open(os.path.join(BASE_DIR, "embeddings.pkl"), "rb") as f:
        return np.array(pickle.load(f))

@st.cache_resource
def load_filenames():
    with open(os.path.join(BASE_DIR, "filenames.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model

feature_list = load_embeddings()
filenames = load_filenames()
model = load_model()

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: bold;
    color: #ff6347;
    text-align: center;
    margin-bottom: 20px;
}
.subtitle {
    font-size: 24px;
    color: #4682b4;
    text-align: center;
    margin-bottom: 40px;
}
</style>

<div class="title">Fashion Recommender System</div>
<div class="subtitle">By Rashid Patel</div>
""", unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS
# -----------------------------
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception:
        return None

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="euclidean"
    )
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    saved_path = save_uploaded_file(uploaded_file)

    if saved_path:
        display_image = Image.open(saved_path)
        st.image(display_image, caption="You provided this item")

        features = feature_extraction(saved_path, model)
        indices = recommend(features, feature_list)

        st.subheader("You might also like these:")
        st.snow()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])
    else:
        st.error("File upload failed. Please try again.")


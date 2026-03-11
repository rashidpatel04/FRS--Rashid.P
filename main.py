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
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------------
# Load embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    with open(os.path.join(BASE_DIR, "embeddings.pkl"), "rb") as f:
        return np.array(pickle.load(f))


# -----------------------------
# Load filenames
# -----------------------------
@st.cache_resource
def load_filenames():
    with open(os.path.join(BASE_DIR, "filenames.pkl"), "rb") as f:
        return pickle.load(f)


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
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

.title{
font-size:42px;
font-weight:bold;
text-align:center;
color:#ff6347;
}

.subtitle{
text-align:center;
font-size:22px;
margin-bottom:30px;
color:#4682b4;
}

</style>

<div class="title">Fashion Recommender System</div>
<div class="subtitle">By Rashid Patel</div>

""", unsafe_allow_html=True)


# -----------------------------
# Save uploaded file
# -----------------------------
def save_uploaded_file(uploaded_file):

    try:

        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(file_path,"wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    except:
        return None


# -----------------------------
# Feature extraction
# -----------------------------
def feature_extraction(img_path,model):

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)

    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img,verbose=0).flatten()

    normalized_result = result / norm(result)

    return normalized_result


# -----------------------------
# Recommendation
# -----------------------------
def recommend(features,feature_list):

    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="euclidean"
    )

    neighbors.fit(feature_list)

    distances,indices = neighbors.kneighbors([features])

    return indices


# -----------------------------
# Display image
# -----------------------------
def show_image(filename):

    image_path = os.path.join(IMAGE_DIR, filename)

    if os.path.exists(image_path):

        st.image(image_path,width=150)

    else:

        st.warning(f"Missing image: {filename}")


# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)


if uploaded_file is not None:

    saved_path = save_uploaded_file(uploaded_file)

    if saved_path:

        st.image(Image.open(saved_path),caption="You provided this item")

        features = feature_extraction(saved_path,model)

        indices = recommend(features,feature_list)

        st.subheader("You might also like these:")

        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            show_image(filenames[indices[0][1]])

        with col2:
            show_image(filenames[indices[0][2]])

        with col3:
            show_image(filenames[indices[0][3]])

        with col4:
            show_image(filenames[indices[0][4]])

        with col5:
            show_image(filenames[indices[0][5]])

    else:

        st.error("File upload failed.")

import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Fashion Recommender | Rashid Patel",
    page_icon="👕",
    layout="wide"
)

# -------------------------
# PATHS
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------

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


# -------------------------
# UI HEADER
# -------------------------

st.markdown("""
<style>

.main-title{
font-size:45px;
font-weight:bold;
text-align:center;
color:#ff4b4b;
}

.sub-title{
font-size:22px;
text-align:center;
color:gray;
margin-bottom:30px;
}

.card{
background:#fafafa;
padding:20px;
border-radius:12px;
}

</style>

<div class="main-title">Fashion Recommender System</div>
<div class="sub-title">AI Powered Fashion Discovery • Built by Rashid Patel</div>

""", unsafe_allow_html=True)

# -------------------------
# FEATURE EXTRACTION
# -------------------------

def feature_extraction(img_path):

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img)

    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result / norm(result)


# -------------------------
# RECOMMEND
# -------------------------

def recommend(features):

    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="euclidean"
    )

    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return distances, indices


# -------------------------
# FILE UPLOAD
# -------------------------

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg","jpeg","png"]
)


if uploaded_file is not None:

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Uploaded Item")

    st.image(file_path, width=300)

    with st.spinner("Analyzing fashion style..."):

        features = feature_extraction(file_path)

        distances, indices = recommend(features)

    st.subheader("Recommended Products")

    cols = st.columns(5)

    for i in range(1,6):

        image_name = filenames[indices[0][i]]
        image_path = os.path.join(IMAGE_DIR, image_name)

        similarity = 1 - distances[0][i]

        with cols[i-1]:

            st.image(image_path, use_column_width=True)

            st.caption(f"Similarity: {similarity:.2f}")

            if st.button(f"View {i}"):

                st.session_state["selected"] = image_path


# -------------------------
# SELECTED IMAGE PREVIEW
# -------------------------

if "selected" in st.session_state:

    st.divider()

    st.subheader("Product Preview")

    st.image(st.session_state["selected"], width=400)

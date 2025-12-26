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

# Load precomputed feature vectors and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the ResNet50 model pre-trained on ImageNet
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a Sequential model that adds a GlobalMaxPooling2D layer to ResNet50
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Streamlit app title with custom styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #ff6347; /* Tomato color */
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        color: #4682b4; /* Steel blue color */
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
    <div class="title">Fashion Recommender System</div>
    <div class="subtitle">By Rashid Patel</div>
""", unsafe_allow_html=True)

# Function to save uploaded files to the 'uploads' directory
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function to extract features from an image using the model
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to find the nearest neighbors of the extracted features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload interface
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image with a label
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="You provided this item")

        # Extract features from the uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Find the nearest neighbors based on the extracted features
        indices = recommend(features, feature_list)

        # Display the recommendations with snow animation
        st.write("You might also like these:")

        # Trigger snow animation
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
        st.header("Some error occurred in file upload")

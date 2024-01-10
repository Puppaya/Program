import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import chardet

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
model_path = os.path.join(script_dir, "keras_model.h5")
model = load_model(model_path, compile=False)

# Load the labels with proper encoding detection
labels_path = os.path.join(script_dir, "labels.txt")
with open(labels_path, "rb") as file:
    result = chardet.detect(file.read())
    encoding = result["encoding"]

with open(labels_path, "r", encoding=encoding) as file:
    class_names = file.readlines()

# Function to preprocess the image
def preprocess_image(image_path):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data

# Streamlit app
st.title("ແບບຈຳລອງປັນຍາປະດິດຈຳແນກປະເພດຢາ")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    input_data = preprocess_image(uploaded_file)

    # Predicts the model
    prediction = model.predict(input_data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display prediction and confidence score
    if confidence_score < 0.75:
        st.warning("ບໍ່ພົບຂໍ້ມູນ")
    else:   
        st.write(f"Class: {class_name.strip()}")
        st.write(f"Confidence Score: {confidence_score}")
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
  

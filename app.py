import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(r"artifacts/training/model.h5")

st.title("Chicken Disease Classification")

uploaded_image = st.file_uploader("Upload a chicken image", type=["jpg", "png"])

if uploaded_image is not None:

    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")
    image = image.resize((224, 224))  
    image_array = np.array(image) / 255.0 
    image_array = np.expand_dims(image_array, axis=0)  

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    result = None
    if predicted_class == '0':
        result = "Positive"
    else:
        result = "Negative"    
    st.write(f"Disease detected: {result}")


#https://drive.google.com/file/d/1synPljrV3ooPhZJqHkO9oCu4wxi3KV_K/view?usp=drive_link
#1synPljrV3ooPhZJqHkO9oCu4wxi3KV_K
#https://drive.google.com/uc?id=1synPljrV3ooPhZJqHkO9oCu4wxi3KV_K
#MODEL_PATH = "vgg19_model_2.h5"
#MODEL_URL = "https://drive.google.com/uc?id=1ABCdEfGhiJklMnOPqrS"
#https://drive.google.com/file/d/1synPljrV3ooPhZJqHkO9oCu4wxi3KV_K/view?usp=sharing
#pip install gdown
#import gdown

#def load_model():
    #with st.spinner("Loading Prediction model..."):
        #model = tf.keras.models.load_model("vgg19_model_2.h5")
    #return model

#model = load_model()
#st.success("Model Loaded Successfully")


import time
from tensorflow.keras.models import load_model
import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuring the streamlit page
st.set_page_config(page_title="Manamuz Group", layout="centered")

success_holder = st.empty()


MODEL_PATH = "vgg19_model_2.h5"
MODEL_URL = "https://drive.google.com/uc?id=1synPljrV3ooPhZJqHkO9oCu4wxi3KV_K"
@st.cache_resource

def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_model_from_drive()
success_holder.success("Model Loaded Successfully")
time.sleep(0.5)
success_holder.empty()



#the header
st.title("ManamuzGroup")
st.markdown("Upload a bell pepper leaf image and click **Predict** to check if it's healthy or diseased.")

# Uploading the picture
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    # To make the user to view the uploaded image in a reduced size
    display_image = image.resize((128, 128))
    st.image(display_image, caption='Uploaded Bell_Pepper', use_container_width=False)

    # Prediction section using the prediction button
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Performing prediction using the model
            prediction = model.predict(img_array)[0]
            class_names = ["Healthy", "Unhealthy"]
            pred_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Displaying the prediction result
            st.markdown(f"### Prediction: **:red[{pred_class}]**")
            st.markdown(f"### Confidence: **{confidence:.2f}%**")

            # Showing chatbot button only if the leaf is Unhealthy
            if pred_class == "Healthy":
                st.markdown('You are Doing a Great Job Farmer! Keep it On!')
            if pred_class == "Unhealthy":
                if st.button("Chat with Virtual Agronomist"):
                    st.info("Agronomist chat feature coming soon...")



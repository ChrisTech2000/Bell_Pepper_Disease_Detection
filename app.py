
#llama-index==0.12.48
#langchain==0.3.26
#llama_cpp_python==0.3.12
#import tensorflow as tf

import time
#from tensorflow.keras.models import load_model
import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

#LLaMA Imports
#from llama_index.core import VectorStoreIndex, ServiceContext
#from langchain.schema import SystemMessage, HumanMessage, AIMessage






if isinstance(st.session_state, str):
    st.session_state.clear()

# Configuring the streamlit page
st.set_page_config(page_title="Manamuz Group", layout="centered")

#success_holder = st.empty()

#Defining session state tohandle navigation
if "page" not in st.session_state:
    st.session_state.page = "main"


#navigating to chat page
if st.session_state.page == "chat":
    st.title("Talk to Virtual Agronomist")
    st.markdown("Ask your Question About your Bell Pepper Crop")

    '''LLAMA_MODEL_PATH = "llama-2-7b-chat.Q2_K.gguf"
    LLAMA_MODEL_URL = "https://drive.google.com/uc?id=1pmzmY3vuSkSeFDptJryHjzUoXyHSYcS3"


    @st.cache_resource
    def download_llama_model():
        if not os.path.exists(LLAMA_MODEL_PATH):
            with st.spinner("Downloading LLaMA model ... (this might take a while, exercise patient)"):
                gdown.download(LLAMA_MODEL_URL, LLAMA_MODEL_PATH, quiet=False)
        return LLAMA_MODEL_PATH
    
    llama_model_path = download_llama_model()


    @st.cache_resource
    def select_llm():
        return LlamaCPP(model_path=llama_model_path, temperature=0.1, max_new_tokens=500, context_window=3900, generate_kwargs={}, model_kwargs={"n_gpu_layers":1}, verbose=True)
    

    def init_message():
        clear_button = st.sidebar.button("Clear Conversation", key = "clear")

        if clear_button:
            st.session_state.pop("messages", None)
            st.rerun()

        if clear_button or "messages" not in st.session_state:
            st.session_state.messages =[SystemMessage(content="You are a helpful AI assistant. Reply Your Answer in a markdown format.")]


    def get_answer(llm, messages) -> str:
        response = llm.complete(messages)
        return response.text
    

    llm = select_llm()
    init_message()


    if user_input := st.chat_input("Input Your Observation"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Virtual Agronomist is typing"):
            answer = get_answer(llm, user_input)
        st.session_state.messages.append(AIMessage(content=answer))


    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


    if st.button("Back to App"):
        st.session_state.page = "main"
        st.rerun()'''


else:
    
    MODEL_PATH = "vgg19_model_2.h5"
    MODEL_URL = "https://drive.google.com/uc?id=1synPljrV3ooPhZJqHkO9oCu4wxi3KV_K"
    @st.cache_resource

    def load_model_from_drive():
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading model..."):
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        return load_model(MODEL_PATH)


#Leting the load ed successfully show once
    if "model_loaded" not in st.session_state:
        model = load_model_from_drive()
        st.session_state.model_loaded = True
        st.markdown(
            "<h4 style='color:green;'>Model Loaded Successfully</h4>",
            unsafe_allow_html=True
        ) 
    else:
        model = load_model(MODEL_PATH)



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

                #saving the state
                st.session_state.predicted_class = pred_class
                st.session_state.prediction_confidence = confidence

                st.rerun()

        if "predicted_class" in st.session_state:
            pred_class = st.session_state.predicted_class
            confidence = st.session_state.prediction_confidence

            # Displaying the prediction result
            st.markdown(f"### Prediction: **:red[{pred_class}]**")
            st.markdown(f"### Confidence: **{confidence:.2f}%**")

            if pred_class == "Healthy":
                st.markdown('You are Doing a Great Job, Farmer! Keep it On!')
            elif pred_class == "Unhealthy":
                st.markdown("Your Plant Appears Unhealthy")
                time.sleep(1.5)

                with st.spinner("Redirecting You to a Virtual Agronomist"):
                    time.sleep(1.5)
                    st.session_state.page = "chat"
                    st.rerun()
  



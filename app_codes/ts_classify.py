## PRE REQUISITE:#########################################
# In conda or cmd: pip install streamlit
# To run the py code: streamlit run prod_recog.py
###########################################################

## Import Libraries
import streamlit as st
import cv2
import numpy as np
from skimage import color
from skimage import io
import pandas as pd
import tensorflow as tf


def model_id_find(radio_sel):
    mod_id=0
    if radio_sel == 'TensorFlow/Keras':
        mod_id = 0
    elif radio_sel == 'PyTorch':
        mod_id = 1
    return mod_id


def mod_select(id):
    model_name = 'model'+str(id)+'.h5'
    return tf.keras.models.load_model(model_name, compile = False)


# loading datasource for link fetching
ds = pd.read_csv('ds.csv')

# Header/Banner
# st.image("header.png")

# radio button for categories
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
model_selection_button = st.radio("Select Model",('TensorFlow/Keras','PyTorch'))

# File upload button
file = st.file_uploader(" ", type=["jpg", "png"])

# 2 Columns
col1, col2 = st.beta_columns([4,4])

#=============================================================================
# Image Recognition
#=============================================================================
if file is None:
    st.text("Please upload an image file")
else:
    with col1:
        # Load the model
        model_id= model_id_find(model_selection_button)
        model = mod_select(model_id)
        
        # Load and display input image
        img = io.imread(file)
        st.image(img, width=300)
        
        # Pre-process input image
        img = np.array(img)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscale
        img = cv2.equalizeHist(img) #equalize
        img = img/255 #normalize
        img = img.reshape(1, 32, 32,1)
   
    with col2: 
        
        # printing prediction output
        prediction = np.max(model.predict_classes(img))
        max = np.max(model.predict(img))
        st.write("Image detected:")
        category_id = prediction

        # checking category and text from ds.csv
        category_name = ds.loc[(ds.id == category_id), 'category'].values[0]
        category_text = ds.loc[(ds.id == category_id), 'text'].values[0]
        
        st.markdown(str("__"+category_name+"__"))
        st.write(round(max * 100,2),"%"," match")
        st.write("\n\n")
        # st.write("Text to speech:")
        # st.markdown(str("__"+category_text+"__"))
        

        # Text to Speech
        from bokeh.models.widgets import Button
        from bokeh.models import CustomJS

        tts_button = Button(label="Speak", width=100)

        tts_button.js_on_event("button_click", CustomJS(code=f"""
            var u = new SpeechSynthesisUtterance();
            u.text = "{category_text}";
            u.lang = 'en-US'; 
            speechSynthesis.speak(u);
            """))
        
        st.bokeh_chart(tts_button)





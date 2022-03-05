import tensorflow
from tensorflow import keras
from keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2

image = Image.open('caliche.jpeg')
st.image(image, width=100)
st.title('Metal Sheet Defects Detection')
#put the path of the hdf5 file as filepath
filepath = 'updated_model_4_small_data.h5'
# Load the model
model = load_model(filepath, compile = True)


uploaded_file = st.file_uploader("",type="jpg")
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR", caption='Uploaded Image')
    image_npy=np.array(opencv_image)

    import numpy as np
    dim=(256,1600)
    X_test = np.empty((1, *dim, 3))
    X_test[0,] = image_npy
    predictions=model.predict(X_test)
    ans=(np.argmax(predictions[0]))
    if ans==1:
        st.markdown('Defected')
    else:
        st.markdown('Not defected')

    

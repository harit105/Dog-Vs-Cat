import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow import keras

st.title("Dogs Vs Cats - Image Classifier")
upload = st.sidebar.file_uploader(label='Upload the Image')
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  model = keras.models.load_model('Dogs_vs_cats.hdf5')
  if st.sidebar.button('Predict'):
    st.sidebar.write("Result:")
    x = cv2.resize(opencv_image,(160,160))
    x = np.expand_dims(x,axis=0)
    y = model.predict(x)
    if y > 0:
      st.success('It is a DOG!!!')
    else:
      st.success('It is a CAT!!!')
      
      

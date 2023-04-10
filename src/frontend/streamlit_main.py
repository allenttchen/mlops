import os
import cv2
import requests
import json

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Configs
MODEL_INPUT_SIZE = 28
CANVAS_SIZE = MODEL_INPUT_SIZE * 8

API_URL = "http://127.0.0.1:5000/"
PREDICT_URL = os.path.join(API_URL, "predict")

st.title("Single Digit Recognition")
st.text("Draw a digit in the canvas below")
canvas_result = st_canvas(
    fill_color="black",  # Fixed fill color with some opacity
    stroke_color="black",
    stroke_width=15,
    update_streamlit=True,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas",
)
#uploaded_image = st.file_uploader("Upload a number image:", type=["png", "jpg"])
if st.button("Predict"):
    predicting = st.text("Predicting...")
    # Scale down image to the model input size
    RGBA_img = canvas_result.image_data.astype("float32")
    grayscale_img = np.mean(RGBA_img, axis=2)
    img = cv2.resize(
        grayscale_img,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    json_img = json.dumps({"input_image": img.tolist()})

    # Rescaled image upwards to show
    #img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # img_rescaled = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
    # st.write("Model input")
    # st.image(img_rescaled, clamp=True)

    # Post model output request
    try:
        response = requests.post(
            url=PREDICT_URL,
            json=json_img,
        )
        if response.ok:
            model_result = response.json()
            st.markdown(f"You drew the number {model_result['result']}!")
            predicting.text("Predicting...Done")
        else:
            st.write("Some error occurred")
    except ConnectionError as e:
        st.write("Couldn't reach backend")

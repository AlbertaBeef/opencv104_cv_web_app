# Importing the libraries.
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from utils import *

def prediction(net):   
    # Draw or clear?
    #drawing_mode = st.checkbox("Draw or clear?",True)
    drawing_mode = "freedraw"

    # Create a canvas component
    #image_data = st_canvas(
    #    15, '#FFF', '#000', height=280,width=280, drawing_mode=drawing_mode, key="canvas"
    #)
    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color='#FFF',
        background_color='#000',
        height=280,width=280, 
        drawing_mode=drawing_mode, 
        key="canvas"
    )

    # Predicting the image
    if canvas_result.image_data is not None:
        if st.button('Predict'):
            # Model inference
            image = canvas_result.image_data;
            #print("[INFO] image shape = ",image.shape)
            digit, confidence = predictDigit(image,net)
            print('[INFO]',' Recognized Digit: {}'.format(digit),' Confidence: {:.2f}'.format(confidence))
            st.write('Recognized Digit: {}'.format(digit))
            st.write('Confidence: {:.2f}'.format(confidence))

def main():
    # Load Digit Recognition model
    net = cv2.dnn.readNetFromONNX('model.onnx')
    
    st.title("Digit Recognizer")
    st.write("\n\n")
    st.write("Draw a digit below and click on Predict button")
    #st.write("\n")
    #st.write("To clear the digit, uncheck checkbox, double click on the digit or refresh the page")
    #st.write("To draw the digit, check the checkbox")

    prediction(net)

if __name__ == '__main__':
    main()

import streamlit as st
from PIL import Image
import random
 # Page Title 
 # st.title("Vehicle Classification App")
 #  # Description
 #  st.write("Upload a vehicle image to classify it as Car, Bus, or Truck.")
 #  # Upload Image
 #  uploaded_file = st.file_uploader( "Choose a vehicle image", type=["jpg", "jpeg", "png"] )
 #  # Vehicle Classes classes = ["Car", "Bus", "Truck"] 
 # # If image uploaded if uploaded_file is not None:
 #  # Open image image = Image.open(uploaded_file) 
 # # Show image st.image( image, caption="Uploaded Image", use_container_width=True ) 
 # # Predict Button if st.button("Predict Vehicle"):
 #  # Demo Prediction prediction = random.choice(classes)
 #  # Show Result st.success(f"Prediction Result: {prediction}") 
 # # Additional Message 
 # if prediction == "Car": 
 # st.info("This vehicle is classified as a Car.")
 #  elif prediction == "Bus": 
 # st.info("This vehicle is classified as a Bus.") 
 # elif prediction == "Truck":
 #  st.info("This vehicle is classified as a Truck.")
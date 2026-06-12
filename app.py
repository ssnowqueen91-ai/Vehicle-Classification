import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load Model
model = load_model("vehicle_classification_model.keras")

# Class Names
class_names = ["bus", "cars", "motorcycle"]
st.title("Vehicle Classification App")

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess Image
    img = image.resize((180, 180))  
    img = np.array(img)
    #img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    st.write("Raw prediction:", prediction)
    st.write("Shape:", prediction.shape)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
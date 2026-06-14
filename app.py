import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import urllib.request

@st.cache_resource
def load_my_model():
    model_path = "vehicle_classification_model.keras"
    
    if not os.path.exists(model_path):
        FILE_ID = "1n_KfcHlRHBm9xKC8bynnVLIwIy9VbwhV"
        URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}"
        
        with st.spinner("Downloading AI Model (45MB)... Please wait..."):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(URL, model_path)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# Class Names
class_names = ["bus", "cars", "motorcycle"]
# --- Sidebar categories display ---
with st.sidebar:
    st.header("🚘 Supported Vehicles")
    st.write("Our AI Model can classify the following vehicle types:")
    
    st.markdown("""
    * 🚌 **Bus** 
    * 🚗 **Cars** 
    * 🏍️ **Motorcycle**
    """)
    
    st.divider() 
    st.caption("Vehicle Classification Project v1.0")
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
    import scipy
    import numpy as np
    import pandas as pd
    exp_preds = np.exp(prediction[0] - np.max(prediction[0]))
    probabilities = exp_preds / exp_preds.sum()
    chart_data = pd.DataFrame(
        probabilities,
        index=class_names,
        columns=["Probability"]
    )

    #  Bar Chart 
    st.write("📊 **Prediction Probability:**")
    st.bar_chart(chart_data)

    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence_score = probabilities[predicted_index]*100
    
    st.success(f"🎉 **Result: This is a {predicted_class.upper()}!** (Confidence: {confidence_score:.2f]}%)")
    st.write("Raw prediction:", prediction)
    st.write("Shape:", prediction.shape)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction:{predicted_class}")
import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

st.title("🍎 Fruit & Vegetable Classifier")
st.write("Upload an image and get instant prediction using AI 🤖")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    return load_model("fruit_model_mobilenetv2.h5")

model = load_my_model()

# ---------------- CLASS NAMES ----------------
train_path = "dataset/train"
class_names = sorted(os.listdir(train_path))

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # Spinner while processing
    with st.spinner("🔍 Analyzing image... Please wait"):
        
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

    # ---------------- RESULT UI ----------------
    st.success("Prediction completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🧠 Prediction")
        st.markdown(f"## {predicted_class}")

        st.subheader("📊 Confidence")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}% confidence")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with AI + Deep Learning 🚀")
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# =============================
# Load Model
# =============================
model = tf.keras.models.load_model("pneumonia_model_final.keras")

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# =============================
# Sidebar
# =============================
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### 🧠 Project Title
AI-Based Pneumonia Detection Using Deep Learning  

### 🏗 Model Architecture
MobileNetV2 (Transfer Learning)

### 📊 Final Test Accuracy
94.7%   *(Update with your real accuracy)*

### 👩‍💻 Project Team

**Ghadeer Ahmad Ghatasha**
- Application Development & Model Implementation  
- Literature Review & Medical Background Research  
- Documentation & Report Writing  

**Rasha Nayef Almashni**
- Literature Review & Medical Background Research  
- Documentation & Report Writing

**Rana Fakhri Shalalda**
- Literature Review & Medical Background Research  
- Documentation & Report Writing 

### 👨‍🏫 Supervisor
Dr. Bassam Arqoub  

### 📅 Year
2026
""")

# =============================
# Main Title
# =============================
st.title("🩺 AI-Based Pneumonia Detection System")
st.markdown("### Chest X-ray Classification using Deep Learning")

st.markdown(
    "Developed as a Graduation Project in Artificial Intelligence."
)

st.write(
    "Upload a chest X-ray image and the AI model will classify it."
)

st.info(
    "⚠️ This system is designed to assist medical professionals in diagnosis. "
    "It is NOT intended to replace clinical judgment or professional medical decisions."
)

st.divider()

# =============================
# Upload Image
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width="stretch")

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.divider()

    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing image using AI model..."):
            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])

        st.subheader("📊 Prediction Result")

        # =============================
        # 3-Level Decision System
        # =============================

        if confidence >= 0.75:
            st.error("🦠 Pneumonia Detected")
            st.write(f"Confidence: **{confidence * 100:.2f}%**")
            st.progress(int(confidence * 100))

        elif confidence <= 0.35:
            st.success("✅ Normal")
            normal_conf = 1 - confidence
            st.write(f"Confidence: **{normal_conf * 100:.2f}%**")
            st.progress(int(normal_conf * 100))

        else:
            st.warning("⚠️ Uncertain – Requires Medical Review")
            st.write(f"Model Output: **{confidence * 100:.2f}%**")
            st.progress(int(confidence * 100))

    if st.button("🔄 Reset"):
        st.rerun()

st.markdown("---")
st.caption("Graduation Project | Deep Learning for Medical Image Classification")

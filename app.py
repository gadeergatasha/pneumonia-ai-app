import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time

# =============================
# Constants
# =============================
MODEL_PATH = "pneumonia_model_final_v2.keras"
IMAGE_SIZE = (224, 224)

LAST_CONV_LAYER_NAME = "out_relu"
CAM_THRESHOLD = 0.40
CAM_ALPHA = 0.30

# Decision logic
PNEUMONIA_THRESHOLD = 0.60
NORMAL_THRESHOLD = 0.40

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="PneumoRay",
    page_icon="🩺",
    layout="centered"
)

# =============================
# Load Model
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =============================
# Helper Functions
# =============================
def preprocess_pil_image(image, image_size=IMAGE_SIZE):
    img = image.resize(image_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_gradcam(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def make_gradcam_overlay(
    image,
    model,
    cam_threshold=CAM_THRESHOLD,
    alpha=CAM_ALPHA,
    colormap=cv2.COLORMAP_TURBO
):
    img_array = preprocess_pil_image(image)
    pred_prob = float(model.predict(img_array, verbose=0)[0][0])

    heatmap = get_gradcam(img_array, model, LAST_CONV_LAYER_NAME)
    heatmap[heatmap < cam_threshold] = 0
    heatmap_uint8 = np.uint8(255 * heatmap)

    original_rgb = np.array(image.resize(IMAGE_SIZE)).astype("uint8")
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.resize(
        heatmap_color,
        (original_bgr.shape[1], original_bgr.shape[0])
    )

    overlay_bgr = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return pred_prob, original_rgb, overlay_rgb

# =============================
# Sidebar
# =============================
st.sidebar.title("Project Information")

st.sidebar.markdown("""
### Project Title
Deep Learning-Based Detection of Pneumonia in Pediatric Chest Radiographs  
*A Validation Study Using Cases Collected from Hebron City*

### Model Architecture
MobileNetV2 (Transfer Learning)

### Decision Thresholds
- Pneumonia: `>= 0.60`
- Normal: `<= 0.40`
- Otherwise: `Uncertain`

### Project Team
- Ghadeer Ahmad Ghatasha
- Rasha Nayef Almashni
- Rana Fakhri Shalalda

### Supervisor
Dr. Bassam Arqoub

### Year
2026
""")

# =============================
# Main UI
# =============================
st.title("🩺 PneumoRay")
st.markdown("### Deep Learning-Based Detection of Pneumonia in Pediatric Chest Radiographs")
st.markdown("Developed as a Graduation Project in Medical Imaging.")

st.info(
    "⚠️ This system is designed to assist medical professionals in diagnosis. "
    "It is not intended to replace clinical judgment or professional medical decisions."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.divider()

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image using AI model..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            pred_prob, original_rgb, gradcam_rgb = make_gradcam_overlay(image, model)

        st.subheader("Prediction Result")

        if pred_prob >= PNEUMONIA_THRESHOLD:
            st.error("Pneumonia Detected")
            st.write(f"Pneumonia Probability: **{pred_prob * 100:.2f}%**")
            st.progress(int(pred_prob * 100))

        elif pred_prob <= NORMAL_THRESHOLD:
            normal_prob = 1 - pred_prob
            st.success("Normal")
            st.write(f"Normal Probability: **{normal_prob * 100:.2f}%**")
            st.progress(int(normal_prob * 100))

        else:
            st.warning("Uncertain - Requires Medical Review")
            st.write(f"Model Output: **{pred_prob * 100:.2f}%**")
            st.write("The image falls inside the uncertainty zone, so the model avoids a confident final diagnosis.")
            st.progress(int(pred_prob * 100))

        st.markdown("### Model Output Details")
        st.write(f"Pneumonia Score: **{pred_prob:.4f}**")
        st.write(f"Normal Score: **{(1 - pred_prob):.4f}**")

        st.markdown("### Grad-CAM Visualization")
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_rgb, caption="Original Resized Image", use_container_width=True)

        with col2:
            st.image(gradcam_rgb, caption="Grad-CAM", use_container_width=True)

        st.caption("Grad-CAM highlights image regions that contributed most to the model's decision.")

    if st.button("Reset"):
        st.rerun()

st.markdown("---")
st.caption("Graduation Project | Deep Learning for Medical Image Classification")
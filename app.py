import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime

# ====================== CONFIG ======================
st.set_page_config(page_title="Rice Disease Detector", page_icon="🌾", layout="wide")

@st.cache_resource
def load_model():
    model_path = "rice_disease_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file `rice_disease_model.h5` not found! Please train first.")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ["bacterial_leaf_blight", "brownspot", "leaf_smut"]

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# ====================== SIDEBAR HISTORY ======================
st.sidebar.title("📜 Test History")
if st.session_state.history:
    for entry in reversed(st.session_state.history[-10:]):  # show last 10
        st.sidebar.image(entry["image"], use_column_width=True)
        st.sidebar.caption(f"**{entry['disease'].replace('_', ' ').title()}** • {entry['confidence']:.1f}%")
        st.sidebar.caption(entry["time"])
        st.sidebar.divider()
else:
    st.sidebar.info("No tests yet. Upload or capture an image!")

# ====================== MAIN APP ======================
st.title("🌾 Rice Leaf Disease Detector")
st.markdown("**Live camera + AI prediction + history log**")

# Input method
col1, col2 = st.columns([3, 1])
with col1:
    input_method = st.radio("Choose input method:", 
                           ["📤 Upload Image", "📷 Live Camera"], 
                           horizontal=True)

image = None
source_bytes = None

if input_method == "📤 Upload Image":
    uploaded = st.file_uploader("Upload rice leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        source_bytes = uploaded.getvalue()

else:  # Live Camera
    picture = st.camera_input("📸 Point camera at rice leaf and take photo")
    if picture:
        image = Image.open(picture).convert("RGB")
        source_bytes = picture.getvalue()

# ====================== PREDICTION ======================
if image is not None:
    st.image(image, caption="📸 Input Image", use_container_width=True)

    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        with st.spinner("Analyzing with AI..."):
            # Preprocess
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(prediction[0])
            predicted_class = class_names[predicted_idx]
            confidence = float(prediction[0][predicted_idx]) * 100

            # Save to history
            st.session_state.history.append({
                "image": source_bytes,
                "disease": predicted_class,
                "confidence": confidence,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Results
            st.success("✅ Prediction Complete!")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.image(image.resize((300, 300)), caption="Analyzed Leaf")

            with col_b:
                st.subheader("📊 Results")
                st.metric("Predicted Disease", predicted_class.replace("_", " ").title())
                st.metric("Confidence", f"{confidence:.2f}%")

                st.write("**All Probabilities**")
                for name, prob in zip(class_names, prediction[0]):
                    pct = float(prob) * 100
                    st.progress(float(prob), text=f"{name.replace('_', ' ').title()}: **{pct:.1f}%**")

            # TODO: Grad-CAM heatmap will go here in next version
            st.info("🔬 **Disease spot heatmap (Grad-CAM)** coming in next update — it will highlight the exact infected areas with colors.")

else:
    st.info("👆 Choose upload or camera above to start")

st.caption("Built with Streamlit + TensorFlow • Stable version for showcase")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Rice Disease Detector", page_icon="🌾", layout="centered"
)


# Load model once (cached so it doesn't reload every time you interact)
@st.cache_resource
def load_model():
    model_path = "rice_disease_model.h5"
    if not os.path.exists(model_path):
        st.error(
            "❌ Model file `rice_disease_model.h5` not found!\n\nPlease run your **training script** first to create the model."
        )
        st.stop()
    return tf.keras.models.load_model(model_path)


model = load_model()

# Class names (must match your dataset folders in alphabetical order)
class_names = ["bacterial_leaf_blight", "brownspot", "leaf_smut"]

# ====================== UI ======================
st.title("🌾 Rice Leaf Disease Detection")
st.markdown(
    "**Upload a rice leaf image** and the AI will instantly tell you if it has Bacterial Leaf Blight, Brown Spot, or Leaf Smut."
)

# File uploader
uploaded_file = st.file_uploader(
    "📤 Choose a rice leaf photo (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Make sure the leaf is clearly visible",
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # Preprocess image exactly like in your original prediction script
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict button
    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing image with AI..."):
            prediction = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(prediction)
            predicted_class = class_names[predicted_idx]
            confidence = float(prediction[0][predicted_idx]) * 100

            # Show nice results
            st.success("✅ Prediction Complete!")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image.resize((280, 280)), caption="Analyzed Leaf")

            with col2:
                st.subheader("📊 Results")
                st.metric(
                    label="Predicted Disease",
                    value=predicted_class.replace("_", " ").title(),
                )
                st.metric(label="Confidence", value=f"{confidence:.2f}%")

                # Show probability for all classes
                st.write("**All Class Probabilities**")
                for name, prob in zip(class_names, prediction[0]):
                    prob_pct = float(prob) * 100
                    st.progress(
                        float(prob),
                        text=f"{name.replace('_', ' ').title()}: **{prob_pct:.1f}%**",
                    )

else:
    st.info("👆 Upload an image above to start the prediction")

# Footer
st.caption("Built with TensorFlow + Streamlit • Model: MobileNetV2 Transfer Learning")

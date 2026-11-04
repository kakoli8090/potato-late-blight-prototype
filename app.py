
import os
import tempfile

import streamlit as st
import numpy as np
import cv2
import pandas as pd

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------
# CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Potato Late Blight Analysis",
    page_icon="🥔",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "03_Models", "best_disease_model.keras"
)

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]


# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_disease_model():
    return load_model(MODEL_PATH, compile=False)


model = load_disease_model()


# -----------------------------
# DISEASE PREDICTION
# -----------------------------

def predict_disease(image_path):

    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    return {
        "predicted_disease": CLASS_NAMES[predicted_index],
        "confidence": float(predictions[0][predicted_index] * 100),
        "probabilities": {
            name: float(probability * 100)
            for name, probability in zip(CLASS_NAMES, predictions[0])
        }
    }


# -----------------------------
# SEVERITY ESTIMATION
# -----------------------------

def estimate_severity(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    # GrabCut leaf detection
    mask = np.zeros(image.shape[:2], np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    height, width = image.shape[:2]

    rect = (5, 5, width - 10, height - 10)

    cv2.grabCut(
        image, mask, rect,
        bg_model, fg_model,
        5, cv2.GC_INIT_WITH_RECT
    )

    leaf_mask = np.where(
        (mask == 2) | (mask == 0), 0, 1
    ).astype(np.uint8)

    # Keep largest connected component
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        leaf_mask, 8
    )

    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask = (labels == largest).astype(np.uint8)

    # HSV mask
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hsv_mask = cv2.inRange(
        hsv,
        np.array([5, 30, 20]),
        np.array([35, 255, 230])
    )

    # Lab mask
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    lab_mask = (
        (a_channel > 125) &
        (b_channel > 125)
    ).astype(np.uint8) * 255

    # HSV OR Lab
    lesion_mask = cv2.bitwise_or(hsv_mask, lab_mask)

    # Restrict to leaf
    lesion_mask = cv2.bitwise_and(
        lesion_mask,
        lesion_mask,
        mask=leaf_mask
    )

    leaf_pixels = np.count_nonzero(leaf_mask)
    lesion_pixels = np.count_nonzero(lesion_mask)

    severity = (
        lesion_pixels / leaf_pixels * 100
        if leaf_pixels > 0 else 0
    )

    # Overlay
    overlay = image.copy()
    overlay[lesion_mask > 0] = (255, 0, 0)

    blended = cv2.addWeighted(
        image, 0.65, overlay, 0.35, 0
    )

    return {
        "severity_percentage": severity,
        "leaf_pixels": int(leaf_pixels),
        "lesion_pixels": int(lesion_pixels),
        "leaf_mask": leaf_mask * 255,
        "lesion_mask": lesion_mask,
        "overlay": blended,
        "original": image
    }




# -----------------------------
# METHOD A: THRESHOLD CLASSIFICATION
# -----------------------------

def classify_severity(severity_percentage):

    if severity_percentage < 10:
        return "Mild"
    elif severity_percentage <= 30:
        return "Moderate"
    else:
        return "Severe"


# -----------------------------
# STREAMLIT INTERFACE
# -----------------------------

st.title("🥔 Potato Late Blight Severity Analysis")

st.write(
    "Upload a potato leaf image to detect the disease "
    "and estimate the affected leaf area."
)

st.warning(
    "This is an experimental research prototype. "
    "Severity estimates are not validated diagnostic measurements."
)

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.subheader("Uploaded Image")

    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Original Potato Leaf",
             width=400)

    if st.button("Analyse Image", type="primary"):

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".jpg"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            image_path = temp_file.name

        try:
            with st.spinner("Analysing image..."):

                disease_result = predict_disease(image_path)

                st.subheader("Disease Classification")

                disease_name = (
                    disease_result["predicted_disease"]
                    .replace("Potato___", "")
                    .replace("_", " ")
                )

                col1, col2 = st.columns(2)

                col1.metric("Predicted Disease", disease_name)

                col2.metric(
                    "Model Confidence",
                    f"{disease_result['confidence']:.2f}%"
                )

                st.write("Class Probabilities")

                probability_df = pd.DataFrame({
                    "Disease": [
                        name.replace("Potato___", "").replace("_", " ")
                        for name in disease_result["probabilities"]
                    ],
                    "Probability (%)": list(
                        disease_result["probabilities"].values()
                    )
                })

                st.dataframe(probability_df, use_container_width=True)

                if disease_name == "Late blight":

                    st.subheader("Percentage-Based Severity Estimation")

                    severity_result = estimate_severity(image_path)

                    # Method A: Class-based severity
                    severity_class = classify_severity(
                        severity_result["severity_percentage"]
                    )

                    st.subheader("Class-Based Severity Estimation (Method A)")

                    st.metric(
                        "Predicted Severity Class",
                        severity_class
                    )

                    st.metric(
                        "Estimated Affected Leaf Area",
                        f"{severity_result['severity_percentage']:.2f}%"
                    )

                    st.write(
                        f"Leaf pixels: {severity_result['leaf_pixels']:,}"
                    )

                    st.write(
                        f"Detected lesion pixels: "
                        f"{severity_result['lesion_pixels']:,}"
                    )

                    st.subheader("Segmentation Results")

                    c1, c2 = st.columns(2)

                    with c1:
                        st.image(
                            severity_result["leaf_mask"],
                            caption="Detected Leaf Mask",
                            clamp=True
                        )

                        st.image(
                            severity_result["lesion_mask"],
                            caption="Detected Lesion Mask",
                            clamp=True
                        )

                    with c2:
                        st.image(
                            severity_result["overlay"],
                            caption="Lesion Overlay",
                            channels="RGB"
                        )

                    # Download result
                    csv_data = pd.DataFrame([{
                        "Predicted Disease": disease_name,
                        "Confidence (%)": disease_result["confidence"],
                        "Estimated Severity (%)":
                            severity_result["severity_percentage"],
                        "Severity Class (Method A)": severity_class,
                        "Leaf Pixels": severity_result["leaf_pixels"],
                        "Lesion Pixels": severity_result["lesion_pixels"]
                    }])

                    st.download_button(
                        "Download Analysis as CSV",
                        data=csv_data.to_csv(index=False),
                        file_name="potato_leaf_analysis.csv",
                        mime="text/csv"
                    )

                else:
                    st.info(
                        "Percentage-based late blight severity "
                        "estimation is only performed for late blight images."
                    )

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

st.divider()

st.caption(
    "Potato Late Blight Research Prototype | "
    "Image-based disease classification and severity estimation"
)

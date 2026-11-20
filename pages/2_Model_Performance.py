
import streamlit as st
import pandas as pd
import os

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Potato Disease Detection - Model Performance")
st.write(
    "Performance evaluation of the MobileNetV2 classifier "
    "and image-based late blight segmentation methods."
)

st.info(
    "Classification and segmentation results are from "
    "separate evaluations. Classification uses 324 test images, "
    "while segmentation uses 10 manually annotated images."
)

# ----------------------------------
# 1. CLASSIFICATION RESULTS
# ----------------------------------

st.header("1. Disease Classification - MobileNetV2")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Accuracy", "98.77%")
col2.metric("Macro Precision", "96.89%")
col3.metric("Macro Recall", "99.11%")
col4.metric("Macro F1-Score", "97.95%")

st.subheader("Classification Evaluation Details")

classification_data = pd.DataFrame({
    "Metric": [
        "Total test images",
        "Correct predictions",
        "Incorrect predictions",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-Score"
    ],
    "Result": [
        "324",
        "320",
        "4",
        "98.77%",
        "96.89%",
        "99.11%",
        "97.95%"
    ]
})

st.dataframe(
    classification_data,
    use_container_width=True,
    hide_index=True
)

# ----------------------------------
# 2. SEGMENTATION RESULTS
# ----------------------------------

st.divider()
st.header("2. Lesion Segmentation Evaluation")

st.write(
    "Four colour-based segmentation approaches were "
    "evaluated against manually annotated CVAT reference masks."
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "segmentation_results.csv"
)

if os.path.exists(CSV_PATH):

    segmentation_data = pd.read_csv(CSV_PATH)

    st.subheader("Segmentation Results Table")

    display_data = segmentation_data.rename(columns={
        "Mean_Dice": "Mean Dice",
        "Mean_IoU": "Mean IoU",
        "Mean_Absolute_Severity_Error_pp":
            "Mean Absolute Severity Error (pp)"
    })

    st.dataframe(
        display_data.round(4),
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------
    # 3. DICE COMPARISON
    # ----------------------------------

    st.subheader("Dice Score Comparison")

    dice_chart = segmentation_data.set_index("Method")[["Mean_Dice"]]
    st.bar_chart(dice_chart)

    # ----------------------------------
    # 4. IOU COMPARISON
    # ----------------------------------

    st.subheader("IoU Score Comparison")

    iou_chart = segmentation_data.set_index("Method")[["Mean_IoU"]]
    st.bar_chart(iou_chart)

    # ----------------------------------
    # 5. SEVERITY ERROR COMPARISON
    # ----------------------------------

    st.subheader("Severity Estimation Error Comparison")

    error_chart = segmentation_data.set_index("Method")[
        ["Mean_Absolute_Severity_Error_pp"]
    ]

    st.bar_chart(error_chart)

    # ----------------------------------
    # 6. BEST BASELINE METHOD
    # ----------------------------------

    best_dice = segmentation_data.loc[
        segmentation_data["Mean_Dice"].idxmax()
    ]

    best_iou = segmentation_data.loc[
        segmentation_data["Mean_IoU"].idxmax()
    ]

    best_error = segmentation_data.loc[
        segmentation_data[
            "Mean_Absolute_Severity_Error_pp"
        ].idxmin()
    ]

    st.divider()
    st.subheader("Best Baseline Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Highest Dice",
        f"{best_dice['Mean_Dice']:.4f}",
        str(best_dice["Method"])
    )

    c2.metric(
        "Highest IoU",
        f"{best_iou['Mean_IoU']:.4f}",
        str(best_iou["Method"])
    )

    c3.metric(
        "Lowest Severity Error",
        f"{best_error['Mean_Absolute_Severity_Error_pp']:.4f}",
        str(best_error["Method"])
    )

    # ----------------------------------
    # 7. RESEARCH INTERPRETATION
    # ----------------------------------

    st.divider()
    st.subheader("Research Interpretation")

    st.write(
        """
        - The MobileNetV2 classifier achieved 98.77% test accuracy.
        - Lab thresholding achieved the highest mean Dice and IoU
          among the four tested baseline methods.
        - Lab thresholding also produced the lowest mean absolute
          severity estimation error.
        - Segmentation performance is preliminary because the
          evaluation used only 10 manually annotated images.
        - Further segmentation optimisation and validation
          are required.
        """
    )

    # ----------------------------------
    # 8. DOWNLOAD RESULTS
    # ----------------------------------

    st.download_button(
        label="Download Segmentation Results CSV",
        data=segmentation_data.to_csv(index=False),
        file_name="segmentation_results.csv",
        mime="text/csv"
    )

else:
    st.error(
        "Segmentation results CSV not found. "
        "Please check the data folder and file name."
    )

st.divider()

st.caption(
    "Potato Late Blight Research Prototype | "
    "Image-Based Disease Classification and Severity Estimation"
)

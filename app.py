
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models import create_resnet18


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroScan | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best.pth"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# CUSTOM BLACK THEME
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #101010;
    color: #eeeeee;
}

.block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

header {
    background: transparent;
}

h1, h2, h3 {
    color: #f5f5f5 !important;
    font-weight: 600 !important;
}

p, label {
    color: #c4c4c4;
}

.brand {
    font-size: 13px;
    letter-spacing: 3px;
    color: #a0a0a0;
    font-weight: 600;
    margin-bottom: 12px;
}

.subtitle {
    color: #999999;
    font-size: 16px;
    margin-top: -8px;
    margin-bottom: 28px;
}

.section-title {
    font-size: 21px;
    font-weight: 600;
    color: #eeeeee;
    margin-top: 25px;
    margin-bottom: 16px;
}

.metric-card {
    background: #1b1b1b;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 22px;
    min-height: 112px;
}

.metric-label {
    color: #999999;
    font-size: 13px;
    margin-bottom: 12px;
}

.metric-value {
    color: #f5f5f5;
    font-size: 23px;
    font-weight: 600;
}

.result-card {
    background: #1b1b1b;
    border: 1px solid #444444;
    border-left: 4px solid #b7b7b7;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.result-label {
    color: #999999;
    font-size: 13px;
    margin-bottom: 7px;
}

.result-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 600;
}

.stButton > button {
    background: #e5e5e5;
    color: #111111;
    border: none;
    border-radius: 8px;
    padding: 10px 25px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #ffffff;
    color: #000000;
}

.stProgress > div > div {
    background-color: #bdbdbd;
}

div[data-testid="stFileUploader"] {
    background: #1b1b1b;
    border: 1px dashed #555555;
    border-radius: 12px;
    padding: 15px;
}

hr {
    border-color: #333333;
}

.footer {
    border-top: 1px solid #333333;
    padding-top: 20px;
    margin-top: 45px;
    color: #777777;
    font-size: 12px;
    text-align: center;
}

.caption-text {
    color: #888888;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = create_resnet18(pretrained=False).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="brand">NEUROSCAN  /  AI RESEARCH SYSTEM</div>',
    unsafe_allow_html=True
)

st.title("Brain MRI Analysis")

st.markdown(
    '<div class="subtitle">'
    'AI-powered four-class brain MRI image classification '
    'with visual explainability.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">01  /  Upload MRI Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Select a brain MRI image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Scan",
        width=350
    )

    st.markdown("")

    if st.button("Analyze MRI", use_container_width=True):

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # ------------------------------------------
        # PREDICTION
        # ------------------------------------------

        with st.spinner("Analyzing MRI image..."):

            with torch.no_grad():

                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)

                confidence, predicted_class = torch.max(
                    probabilities, dim=1
                )

            predicted_index = predicted_class.item()
            predicted_name = CLASS_NAMES[predicted_index]
            confidence_value = confidence.item() * 100

        st.divider()

        # ------------------------------------------
        # PREDICTION RESULT
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">02  /  Prediction Results</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">PREDICTED CLASS</div>
                    <div class="metric-value">{predicted_name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">MODEL CONFIDENCE</div>
                    <div class="metric-value">{confidence_value:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ------------------------------------------
        # CLASS PROBABILITIES
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">03  /  Class Probabilities</div>',
            unsafe_allow_html=True
        )

        for name, probability in zip(
            CLASS_NAMES, probabilities[0]
        ):

            value = probability.item()

            col1, col2 = st.columns([1, 5])

            with col1:
                st.write(f"**{name}**")

            with col2:
                st.progress(value)

            st.caption(f"{value * 100:.2f}%")

        # ------------------------------------------
        # GRAD-CAM
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">04  /  Grad-CAM Explainability</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="caption-text">'
            'Visual representation of the image regions '
            'influencing the model prediction.'
            '</p>',
            unsafe_allow_html=True
        )

        target_layers = [model.layer4[-1]]

        with st.spinner("Generating Grad-CAM visualization..."):

            with GradCAM(
                model=model,
                target_layers=target_layers
            ) as cam:

                targets = [
                    ClassifierOutputTarget(predicted_index)
                ]

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets
                )[0]

        resized_image = image.resize((224, 224))

        rgb_image = (
            np.array(resized_image).astype(np.float32) / 255.0
        )

        cam_overlay = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                resized_image,
                caption="Original MRI",
                use_container_width=True
            )

        with col2:

            st.image(
                cam_overlay,
                caption="Grad-CAM Heatmap Overlay",
                use_container_width=True
            )

        st.caption(
            "Red and yellow regions indicate stronger activation. "
            "Grad-CAM does not provide exact tumor segmentation."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    NEUROSCAN | Brain MRI Classification Research Project<br><br>
    Research and educational prototype only.
    Not intended for clinical diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)

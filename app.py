import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np #img->array for gradcam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget #heatmap only for predicted class

from model import create_model

class_names = [
    "AMD",
    "Catarct",
    "DR",
    "Glaucoma",
    "Hypertensive_Retinopathy",
    "Normal_Fundus",
    "Pathological_Myopia"
]

MODEL_PATH = "retinal_fundus_model.pth"


#load model
@st.cache_resource
def load_model():

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


#image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#the gradcam function
def generate_gradcam(image, image_tensor):
    target_layers = [model.layer4[-1]]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        with torch.enable_grad():
            prediction = model(image_tensor).argmax(dim=1).item() #dim=1 is img dim[1,7]

            grayscale_cam = cam(
                input_tensor=image_tensor,
                targets=[ClassifierOutputTarget(prediction)]
            )[0]

    rgb_image = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0  #get pixel values 0-1 instead of 0-255

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization


#dash
st.set_page_config(
    page_title="Retinal Disease Classifier",
    layout="centered"
)

st.title("Retinal Disease Classifier")

st.write(
    "Upload a retinal fundus image to classify it into one of seven categories."
)

uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Unable to read this file.")
        st.stop()

    st.image(
        image,
        caption="Uploaded retinal image",
        width='stretch'
    )

    if st.button("Classify Image"):

        image_tensor = transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = class_names[
            prediction.item()
        ]

        confidence_percent = confidence.item() * 100

        st.success(
            f"Prediction: **{predicted_class}**"
        )

        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )
        gradcam_img = generate_gradcam(
            image,
            image_tensor
        )

        st.subheader("Grad-CAM Heatmap")

        st.image(
            gradcam_img,
            caption="Grad-CAM",
            width='stretch'
        )

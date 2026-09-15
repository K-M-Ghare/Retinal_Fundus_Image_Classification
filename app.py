import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

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

MODEL_PATH = "fundus_model.pth"


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


#dash
st.set_page_config(
    page_title="Retinal Disease Classifier",
    layout="centered"
)

st.title("Retinal Disease Classifier")

st.write(
    "Upload a retinal fundus image to classify it "
    "into one of seven categories."
)

uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded retinal image",
        use_container_width=True
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

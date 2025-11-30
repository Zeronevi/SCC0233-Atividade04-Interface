import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Classes from the dataset
CLASSES = [
    "Damaged concrete structures",
    "DamagedElectricalPoles",
    "DamagedRoadSigns",
    "DeadAnimalsPollution",
    "FallenTrees",
    "Garbage",
    "Graffitti",
    "IllegalParking",
    "Potholes and RoadCracks"
]


# CNN Model definition (same as in the training file)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 256 x 1 x 1
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes=10).to(device)

    # Try to load the trained weights (you need to upload urbanissuescnn.pth)
    model_path = "urbanissuescnn.pth"
    if os.path.exists('urbanissuescnn.pth'):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("️No trained model found in program folder.")

    model.eval()
    return model, device


# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.title("Urban Issues Image Classifier")
st.markdown("Upload an image of an urban problem to get the predicted category.")

# Load model
model, device = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "JPG", "PNG"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image")

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    #no_grad() for better performance
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        predicted_class = CLASSES[predicted]

    # display results
    st.subheader("Prediction Results")
    st.success(f"**Predicted Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence.item():.2%}")

    #top 3 predictions
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    st.subheader("Top 3 Predictions:")
    for i, idx in enumerate(top3_indices):
        class_name = CLASSES[idx]
        prob = top3_prob[i].item()
        st.write(f"{i + 1}. **{class_name}** ({prob:.2%})")


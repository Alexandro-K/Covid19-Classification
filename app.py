import os
import torch
import streamlit as st

from PIL import Image
from model import SimpleCNN
from torchvision import transforms

# Page Config
st.set_page_config(page_title="Chest X-ray Classification",
                   layout="centered")

st.title("Chest X-ray Classification")
st.warning("""
⚠️ This tool is for research and educational purposes only.
It is not intended for clinical diagnosis.
""")
st.write("Upload an X-ray image to classify Or Try Example Images")

# Example Images
example_images = {"Normal": "images/normal.jpeg",
                  "Covid": "images/covid.jpeg",
                  "Viral Pneumonia": "images/pneumonia.jpeg"}

selected_example = st.selectbox("Choose an example image:",
                                ["None"] + list(example_images.keys()))

# Load Model (Cached)
@st.cache_resource # With this model only be loaded once and used mutliple times
def load_model():
    model = SimpleCNN(num_classes=3) # Initialize the model
    model.load_state_dict(torch.load("Simplecnn_best.pth", map_location="cpu")) # Load the "trained" data
    model.eval() 
    return model

model = load_model()

# Class Labels
classes =  ['Covid', 'Normal', 'Viral Pneumonia']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# File Upload
image = None
uploaded_file = st.file_uploader("Upload X-ray Image",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif selected_example != "None":
    image_path = example_images[selected_example]
    image = Image.open(image_path).convert("RGB")

if image:
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    st.success(f"Prediction: {classes[predicted_class]}")
    st.info(f"Confidence: {probabilities[0][predicted_class].item():.4f}")
    
    # Show all Class Probabilities
    st.subheader("Prediction Probabilities")
    prob_dict = {classes[i]: float(probabilities[0][i])
                 for i in range(len(classes))}
    st.bar_chart(prob_dict)
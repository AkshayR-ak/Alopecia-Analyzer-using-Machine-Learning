import streamlit as st

# Must be the first Streamlit command!
st.set_page_config(page_title="Hair Loss Classification App", layout="wide")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# ----- Additional Imports for GPT-4 Doctor Suggestions -----
import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
import time
import g4f

# Set the event loop policy for Windows
asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

# ----- GPT-4 Functions -----
# def generate_response(user_input):
#     """Generate a response using GPT-4 via g4f."""
#     try:
#         response = g4f.ChatCompletion.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": user_input}],
#             temperature=0.6,
#             top_p=0.9
#         )
#         return response.strip() if response else "Sorry, I didn't understand that."
#     except Exception as e:
#         return f"Error: {e}"

# def generate_doctor_suggestions(condition):
#     """
#     Generate detailed doctor suggestions for the given condition.
#     The response includes:
#       - Why and how the condition occurs
#       - Treatment options and ways to overcome it
#       - Recommended food diet to help manage the condition
#       - Suggestions for nearby hospital locations with Google Maps links
#     """
#     prompt = (
#         f"I have been diagnosed with {condition}. "
#         "Please provide detailed doctor suggestions including: "
#         "- Why this condition occurs and how it develops, "
#         "- Treatment options and ways to overcome it, "
#         "- Recommended food diet to help manage the condition, "
#         "- Suggestions for nearby hospital locations in tamil nadu with Google Maps links for each"
#         " hospital. "
#         "Present the information in a clear and organized manner."
#     )
#     return generate_response(prompt)

#Default doctor response
def generate_doctor_suggestions(condition):
    suggestions = {
        "Alopecia Areata": """
### Causes
- Autoimmune disorder
- Genetics
- Stress

### Treatment
- Corticosteroid injections
- Minoxidil
- Immunotherapy

### Diet
- Eggs
- Fish
- Spinach
- Nuts
- Fruits

### Hospitals in Tamil Nadu
1. Apollo Hospitals, Chennai
   https://maps.google.com/?q=Apollo+Hospitals+Chennai

2. CMC Vellore
   https://maps.google.com/?q=CMC+Vellore

3. SIMS Hospital, Chennai
   https://maps.google.com/?q=SIMS+Hospital+Chennai
"""
    }

    return suggestions.get(
        condition,
        "No information available for this condition."
    )


# ----- Custom CSS Styling -----
st.markdown(
    """
    <style>
    /* Set the background color and font */
    .reportview-container {
        background: #f5f5f5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf, #2e7bcf);
        color: white;
    }
    /* Customize buttons */
    .stButton>button {
        background-color: #2e7bcf;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    /* Customize the main container */
    .block-container {
        padding: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Sidebar -----
st.sidebar.title("About This App")
st.sidebar.info(
    """
    **Alopecia Areata Classification App**  
    This application uses a machine learning model to classify images as either  
    **Alopecia Areata** or **Normal hairs**.  
    Upload one or more images to see predictions with confidence levels.
    
    Additionally, use the Doctor Suggestions feature to get detailed guidance on:
      - Why and how the condition occurs
      - Treatment options and ways to overcome it
      - Recommended food diet
      - Nearby hospital locations (with Google Map )
    """
)

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
1. **Upload Images:** Use the uploader to add one or multiple images.
2. **View Predictions:** Each image will display its predicted label and confidence score.
3. **Probability Chart:** A bar chart shows the probability distribution for each image.
4. **Prediction History:** See a summary table for all processed images.
5. **Doctor Suggestions:** Enter a condition (e.g., "Alopecia Areata") to get detailed doctor advice.
    """
)

# ----- Main Title & Banner -----
st.markdown("<h1 style='text-align: center; color: #2e7bcf;'>Alopecia Areata Classifier</h1>", unsafe_allow_html=True)
# Optionally, add a banner image:
# st.image("banner.jpg", use_column_width=True)

# ----- Device & Transforms -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- Model Loading Function -----
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Loads the checkpoint, rebuilds the model architecture based on the saved model name,
    loads the state dict, and sets the model to evaluation mode.
    """
    checkpoint = torch.load("best_model.pth", map_location=device)
    model_name = checkpoint["model_name"]
    state_dict = checkpoint["state_dict"]
    num_classes = 2  # Update if you have more classes

    if model_name == "ResNet50":
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "DenseNet121":
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Unsupported model type!")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Load the model (cached)
model = load_model()

# ----- File Uploader & Classification -----
uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    predictions = []  # To store prediction details for history
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Load image and display it in a two-column layout
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=uploaded_file.name, use_container_width=True)
            
            # Preprocess the image and predict
            image_tensor = test_transforms(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
            class_names = ["Alopecia Areata", "Normal hairs"]  # Update as needed
            predicted_class = class_names[pred_idx.item()]
            confidence_pct = confidence.item() * 100
            
            with col2:
                st.markdown(f"### Prediction: {predicted_class}")
                st.markdown(f"**Confidence:** {confidence_pct:.2f}%")
                # Plot probabilities as a bar chart
                fig, ax = plt.subplots()
                ax.bar(class_names, probs.cpu().squeeze().numpy(), color=['#2e7bcf', '#66cdaa'])
                ax.set_ylim([0, 1])
                ax.set_ylabel("Probability")
                ax.set_title("Probability Distribution")
                st.pyplot(fig)
            
            # Append prediction to history
            predictions.append({
                "Filename": uploaded_file.name,
                "Prediction": predicted_class,
                "Confidence (%)": f"{confidence_pct:.2f}"
            })
    
    # Display Prediction History
    st.markdown("---")
    st.subheader("Prediction History")
    df_predictions = pd.DataFrame(predictions)
    st.dataframe(df_predictions, use_container_width=True)

# ----- Doctor Suggestions Section -----
st.markdown("---")
st.subheader("Doctor Suggestions")
st.markdown("Enter a condition to get detailed doctor suggestions, including why and how the condition occurs, treatment options, recommended food diet, and nearby hospital locations with Google Maps links.")
condition_input = st.text_input("Enter condition for doctor suggestions", value="Alopecia Areata")
if st.button("Get Doctor Suggestions"):
    with st.spinner("Generating doctor suggestions..."):
         suggestions = generate_doctor_suggestions(condition_input)
         st.markdown(suggestions)

# ----- Footer -----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Batch 25 | &copy; 2025</p>", unsafe_allow_html=True)

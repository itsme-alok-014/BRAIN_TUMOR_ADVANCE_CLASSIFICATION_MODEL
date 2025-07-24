import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import imutils

# Function to crop brain contour
def crop_brain_contour(image):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image, then perform erosions + dilations to remove small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours and grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        return image
    
    c = max(cnts, key=cv2.contourArea)
    
    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Crop the brain region
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    
    return new_image

# Function to process image for prediction
def process_scan(image, image_size):
    """
    Process the image for model prediction.
    
    Args:
        image: Input OpenCV image
        image_size: Tuple of (width, height)
    
    Returns:
        X: Processed image ready for model input
    """
    X = []
    image_width, image_height = image_size
    
    # Crop the brain and ignore the unnecessary rest part of the image
    image = crop_brain_contour(image)
    
    # Resize image
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    
    # Normalize values
    image = image / 255.
    
    # Convert image to numpy array and append it to X
    X.append(image)
    X = np.array(X)
    
    return X

# Function to analyze the MRI scan and classify tumor type
def analyze_brain_scan(img_array, model_type="cnn"):
    """
    Analyzes brain MRI scan to detect and classify tumor.
    
    Args:
        img_array: Input image array
        model_type: Type of model to use ('cnn' or 'hybrid')
    
    Returns:
        result: Dictionary containing classification results and confidence scores
    """
    IMG_WIDTH, IMG_HEIGHT = (240, 240)
    
    # Process the image
    X = process_scan(img_array, (IMG_WIDTH, IMG_HEIGHT))
    
    # Load the selected model
    if model_type == "cnn":
        model_path = "./Brain_Tumor_Classification_Models/final_cnn_model.keras"
        if not os.path.exists(model_path):
            model_path = "./Brain_Tumor_Detection_Models/cnn-parameters-improvement-01-0.88.keras"
    else:
        model_path = "./Brain_Tumor_Classification_Models/final_hybrid_model.keras"
    
    model = load_model(filepath=model_path)
    
    # Get model prediction
    start_time = time.time()
    prediction = model.predict(X)
    end_time = time.time()
    
    # Define class names
    class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx] * 100
    
    # Create result dictionary
    result = {
        "prediction": class_names[predicted_class_idx],
        "confidence": confidence,
        "all_confidences": {class_names[i]: prediction[0][i] * 100 for i in range(len(class_names))},
        "processing_time": end_time - start_time,
        "model_type": "CNN" if model_type == "cnn" else "Hybrid"
    }
    
    return result

# Function to get result color based on tumor type
def get_result_color(tumor_type):
    if tumor_type == "No Tumor":
        return "green"
    elif tumor_type in ["Glioma", "Meningioma", "Pituitary"]:
        return "red"
    else:
        return "yellow"

# Streamlit UI Layout
# Streamlit UI Layout
st.set_page_config(
    page_title="Advanced Brain Tumor Classification System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling including a big, bold title
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0B3D91;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 500;
    }
    .result-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .tumor-negative {
        color: green;
        font-weight: 600;
    }
    .tumor-positive {
        color: red;
        font-weight: 600;
    }
    .confidence-meter {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .prediction-time {
        font-style: italic;
        color: gray;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar for model selection
st.sidebar.markdown("# Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Classification Model",
    ["CNN Model", "Hybrid Model"],
    help="CNN Model is faster but may be less accurate. Hybrid Model combines multiple architectures for potentially better performance."
)

# Map selection to model type parameter
selected_model = "cnn" if model_type == "CNN Model" else "hybrid"

# Information about models
with st.sidebar.expander("About the Models"):
    st.markdown("""
    **CNN Model**: A standard convolutional neural network trained specifically for brain tumor classification.
    
    **Hybrid Model**: A combination of pretrained models (VGG16 and ResNet50) with a custom CNN, designed to capture different types of features for better classification performance.
    """)

# Information about tumor types
with st.sidebar.expander("About Brain Tumors"):
    st.markdown("""
    **Glioma**: Originating in the glial cells, these can be low or high grade. Glioblastoma is an aggressive type of glioma.
    
    **Meningioma**: These tumors arise from the meninges, the membranes that surround the brain and spinal cord. Most are benign.
    
    **Pituitary**: These tumors develop in the pituitary gland and can affect hormone production.
    """)


# 🧠 Add Main Title Heading
st.markdown('<p class="main-header">🧠 Advanced Brain Tumor Classification System</p>', unsafe_allow_html=True)

st.markdown("""
This application uses deep learning to analyze MRI brain scans and classify them into different types of brain tumors. 
The system can detect and classify between gliomas, meningiomas, pituitary tumors, or non-tumorous conditions.
""")

# File uploader for MRI scan
uploaded_file = st.file_uploader(
    label="Upload MRI Scan (PNG, JPEG, JPG)",
    type=["png", "jpeg", "jpg"],
    help="Upload a brain MRI scan image to analyze"
)

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # Create columns for image and results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">Uploaded MRI Scan</p>', unsafe_allow_html=True)
        st.image(opencv_image, channels="BGR", use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">Analysis</p>', unsafe_allow_html=True)
        
        # Add a "Analyze" button
        if st.button("🔍 Analyze Scan", key="analyze_button"):
            with st.spinner("Processing MRI scan..."):
                # Get analysis results
                results = analyze_brain_scan(opencv_image, selected_model)
                
                # Display the result with appropriate formatting
                result_color = get_result_color(results["prediction"])
                
                # Show prediction
                st.markdown(f'<p class="result-header">Prediction Result:</p>', unsafe_allow_html=True)
                
                # Show the predicted class with appropriate styling
                if results["prediction"] == "No Tumor":
                    st.markdown(f'<p class="tumor-negative">✅ {results["prediction"]}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="tumor-positive">⚠️ {results["prediction"]}</p>', unsafe_allow_html=True)
                
                # Show confidence score as a metric and progress bar
                st.markdown('<p class="confidence-meter">Confidence Score:</p>', unsafe_allow_html=True)
                st.metric(label="", value=f"{results['confidence']:.1f}%")
                st.progress(float(results['confidence']/100))
                
                # Show all class confidences
                st.markdown("### All Class Probabilities:")
                
                # Convert confidences to a horizontal bar chart
                classes = list(results["all_confidences"].keys())
                confidences = list(results["all_confidences"].values())
                
                fig, ax = plt.subplots(figsize=(10, 5))
                y_pos = np.arange(len(classes))
                
                # Use different colors for bars
                colors = ['red' if c == results["prediction"] else 'blue' for c in classes]
                
                ax.barh(y_pos, confidences, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(classes)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Confidence (%)')
                
                st.pyplot(fig)
                
                # Model information
                st.markdown('<div class="model-info">', unsafe_allow_html=True)
                st.markdown(f"**Model Used**: {results['model_type']}")
                st.markdown(f"**Processing Time**: {results['processing_time']:.3f} seconds")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional information about the detected tumor type
                if results["prediction"] != "No Tumor":
                    tumor_info = {
                        "Glioma": """
                            **Gliomas** arise from glial cells and can be low or high grade. High-grade gliomas, like glioblastoma, 
                            are aggressive and fast-growing. Treatment typically involves surgery followed by radiation and chemotherapy.
                            """,
                        "Meningioma": """
                            **Meningiomas** develop from the meninges covering the brain and spinal cord. Most are benign and slow-growing.
                            Treatment may include observation, surgery, or radiation therapy depending on size and location.
                            """,
                        "Pituitary": """
                            **Pituitary tumors** form in the pituitary gland and can affect hormone production. 
                            They're usually benign but can cause health problems by producing excess hormones or pressing on nearby structures.
                            Treatment options include medication, surgery, or radiation therapy.
                            """
                    }
                    
                    with st.expander(f"About {results['prediction']} Tumors"):
                        st.markdown(tumor_info[results["prediction"]])
                
            
else:
    # Display sample images or instructions when no file is uploaded
    st.info("Please upload an MRI scan image to begin analysis.")



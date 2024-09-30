import streamlit as st
from PIL import Image
import cv2
import numpy as np
from _utils import pil_to_bytes, bytes_to_base64, get_initial_image_as_bytes, common_markdown

def detect_faces(image, scale_factor):
    # Convert image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=5, minSize=(30, 30))
    # Clone the input image to draw rectangles on
    output_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return output_image, len(faces)

# Set layout
st.set_page_config(layout="wide")

# Set title and instructions
st.title("Interactive Human Face Detection")
st.write("## Upload an image for face detection.")

initial_img = get_initial_image_as_bytes("human_face")

# Set image uploader and slider for scale factor
uploaded_img = st.sidebar.file_uploader("Upload an image (jpg, png)", type=["jpg", "png"])
scale_factor = st.sidebar.slider("Scale Factor (smaller values increase sensitivity and computational cost)", 1.01, 1.5, 1.1, 0.01)

# Initial Display
col1, col2 = st.columns(2)
col1.header("Original Image")
col2.header("Face Detection Image")

# Use uploaded image, if present, otherwise use initial image
img_to_use = initial_img if uploaded_img is None else Image.open(uploaded_img)

# Convert image to openCV color space (BGR to RGB)
img_as_array = np.array(img_to_use)[:,:,::-1]

# Face detection
img_detected_faces, num_faces = detect_faces(img_as_array, scale_factor)
detected_faces_img = Image.fromarray(cv2.cvtColor(img_detected_faces, cv2.COLOR_BGR2RGB))


col1.image(img_to_use, use_column_width=True)
col2.image(detected_faces_img, use_column_width=True)
col2.write(f'Number of human faces detected: {num_faces}')

# Calculate the base64 strings for the images
orig_img_bytes = pil_to_bytes(img_to_use)
b64_orig = bytes_to_base64(orig_img_bytes)

processed_img_bytes = pil_to_bytes(detected_faces_img)
b64_processed = bytes_to_base64(processed_img_bytes)

# Create download links
st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_orig}" download="original.png" class="download-button">Download Original Image</a>',
    unsafe_allow_html=True)

st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_processed}" download="processed.png" class="download-button">Download Face Detected Image</a>',
    unsafe_allow_html=True)

# Inject CSS to make the hyperlink looks like a button from Streamlit
st.markdown(common_markdown(), unsafe_allow_html=True)

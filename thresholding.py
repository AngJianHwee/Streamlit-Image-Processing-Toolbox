import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import base64

from _utils import pil_to_bytes, bytes_to_base64, get_initial_image_as_bytes, common_markdown

# Function to convert PIL image to bytes
def pil_to_bytes(image: Image) -> bytes:
    byteIO = io.BytesIO()
    image.save(byteIO, format='PNG')
    return byteIO.getvalue()

# Function to convert bytes to base64
def bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


# Set layout
st.set_page_config(layout="wide")

# Set title and instructions
st.title("Interactive Image Thresholding")
st.write("## Upload an image and adjust the threshold value.")

initial_img = get_initial_image_as_bytes()

# Set image uploader and threshold slider
uploaded_img = st.sidebar.file_uploader("Upload an image (jpg, png)", type=["jpg", "png"], accept_multiple_files=False)
threshold_value = st.sidebar.slider("Set Threshold Value", 0, 255, 127)  # Threshold slider
color_preserve = st.sidebar.checkbox("Preserve color above threshold")  # Checkbox

# Initial Display
col1, col2 = st.columns(2)
col1.header("Original Image")
col2.header("Thresholded Image")

# Use uploaded image, if present, otherwise use initial image
img_to_use = initial_img if uploaded_img is None else Image.open(uploaded_img)

img_array = np.array(img_to_use)

# Threshold in grayscale
gray_img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
_, thresholded_img_array = cv2.threshold(gray_img_array, threshold_value, 255, cv2.THRESH_BINARY)  # Apply thresholding
mask = np.repeat(thresholded_img_array[:, :, np.newaxis], 3, axis=2)

if color_preserve:
    # apply mask to original image in all 3 channels
    thresholded_img_array = np.where(mask, img_array, 0)
    thresholded_img = Image.fromarray(np.uint8(thresholded_img_array))
    
else:
    thresholded_img = Image.fromarray(np.uint8(mask))

col1.image(img_to_use, use_column_width=True)
col2.image(thresholded_img, use_column_width=True)


# Calculate the base64 string for the images
orig_img_bytes = pil_to_bytes(img_to_use)
b64_orig = bytes_to_base64(orig_img_bytes)

processed_img_bytes = pil_to_bytes(thresholded_img)
b64_processed = bytes_to_base64(processed_img_bytes)

# Create download links 
st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_orig}" download="original.png" class="download-button">Download Original Image</a>',
    unsafe_allow_html=True)

st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_processed}" download="thresholded.png" class="download-button">Download Thresholded Image</a>',
    unsafe_allow_html=True)
        

# Inject CSS to make the hyperlink looks like a button from Streamlit
st.markdown(common_markdown(), unsafe_allow_html=True)

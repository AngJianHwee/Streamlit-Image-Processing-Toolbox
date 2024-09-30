import streamlit as st
from PIL import Image
import cv2
import numpy as np

from _utils import pil_to_bytes, bytes_to_base64, get_initial_image_as_bytes, common_markdown

# Set layout
st.set_page_config(layout="wide")

# Set title and instructions
st.title("Interactive Canny Edge Detection")
st.write("## Upload an image and set thresholds for edge detection.")

initial_img = get_initial_image_as_bytes()

# Set image uploader, sliders, and button on sidebar
uploaded_img = st.sidebar.file_uploader("Upload an image (jpg, png)", type=[
                                        "jpg", "png"], accept_multiple_files=False)
lower_threshold = st.sidebar.slider(
    "Set Lower Threshold", 1, 200, 100)  # Lower threshold for Canny
upper_threshold = st.sidebar.slider(
    "Set Upper Threshold", 1, 500, 200)  # Upper threshold for Canny

# Initial Display
col1, col2 = st.columns(2)
col1.header("Original Image")
col2.header("Edge Detection Image")

# Use uploaded image, if present, otherwise use initial image
img_to_use = initial_img if uploaded_img is None else Image.open(uploaded_img)

img_array = np.array(img_to_use.convert('L'))  # convert image to grayscale

# Canny edge detection
edge_img_array = cv2.Canny(img_array, lower_threshold, upper_threshold)
edge_img = Image.fromarray(edge_img_array)

col1.image(img_to_use, use_column_width=True)
col2.image(edge_img, use_column_width=True)

# Calculate the base64 strings for the images
orig_img_bytes = pil_to_bytes(img_to_use)
b64_orig = bytes_to_base64(orig_img_bytes)

proccessed_img_bytes = pil_to_bytes(edge_img)
b64_processed = bytes_to_base64(proccessed_img_bytes)

# Create download links
st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_orig}" download="original.png" class="download-button">Download Original Image</a>',
    unsafe_allow_html=True)

st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_processed}" download="processed.png" class="download-button">Download Edge Detection Image</a>',
    unsafe_allow_html=True)

# Inject CSS to make the hyperlink looks like a button from Streamlit
st.markdown(common_markdown(), unsafe_allow_html=True)

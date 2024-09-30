import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import base64

from _utils import pil_to_bytes, bytes_to_base64, get_initial_image_as_bytes, common_markdown

# Set layout
st.set_page_config(layout="wide")

# Set title and instructions
st.title("Interactive Gaussian Blur")
st.write("## Upload an image and adjust the blur intensity.")


initial_img = get_initial_image_as_bytes()

# Set image uploader, sliders, and button on sidebar
uploaded_img = st.sidebar.file_uploader("Upload an image (jpg, png)", type=[
                                        "jpg", "png"], accept_multiple_files=False)
sigma = st.sidebar.slider("Set Sigma for Gaussian Blur",
                          0.0, 10.0, 2.0)  # Sigma slider
# Kernel size slider (truncate value)
truncate = st.sidebar.slider(
    "Set Kernel Size for Gaussian Blur (Truncate value)", 0.1, 5.0, 2.0)

# Compute kernel size
kernel_size = int(truncate * sigma) if int(truncate *
                                           sigma) % 2 == 1 else int(truncate * sigma) + 1

# Initial Display
col1, col2 = st.columns(2)
col1.header("Original Image")
col2.header("Filtered Image")

# Use uploaded image, if present, otherwise use initial image
img_to_use = initial_img if uploaded_img is None else Image.open(uploaded_img)

img_array = np.array(img_to_use)
blurred_img_array = cv2.GaussianBlur(
    img_array, (kernel_size, kernel_size), sigmaX=sigma)
blurred_img = Image.fromarray(np.uint8(blurred_img_array))

col1.image(img_to_use, use_column_width=True)
col2.image(blurred_img, use_column_width=True)


# calculate the base64 string for the images
orig_img_bytes = pil_to_bytes(img_to_use)
b64_orig = bytes_to_base64(orig_img_bytes)

processed_img_bytes = pil_to_bytes(blurred_img)
b64_processed = bytes_to_base64(processed_img_bytes)

# create download links
st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_orig}" download="original.png" class="download-button">Download Original Image</a>',
    unsafe_allow_html=True)

st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_processed}" download="processed.png" class="download-button">Download Processed Image</a>',
    unsafe_allow_html=True)

# Inject CSS to make the hyperlink looks like a button from Streamlit
st.markdown(common_markdown(), unsafe_allow_html=True)

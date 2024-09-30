import streamlit as st
from PIL import Image
import cv2
import numpy as np
from _utils import pil_to_bytes, bytes_to_base64, get_initial_image_as_bytes, common_markdown

def get_gradient_image(img, kernel_size):    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute gradients along X and Y direction.
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Compute gradient magnitude and direction
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    # Create images for magnitude and angle
    mag_img = np.uint8(255 * grad_mag / np.max(grad_mag))  # Scale to 0-255
    ang_img = np.uint8(255 * (grad_angle + 180) / 360)  # Scaling (-180-180) to (0-255)
    return mag_img, ang_img

# def grid_based_subsampling(mag_img, ang_img, grid_size):
    cell_size_x = mag_img.shape[1] // grid_size 
    cell_size_y = mag_img.shape[0] // grid_size

    subsampled_image = np.zeros((grid_size, grid_size))
    vector_field = np.zeros((grid_size, grid_size, 2))  # Store magnitude and direction

    for i in range(grid_size):
        for j in range(grid_size):
            cell_magnitude = np.mean(mag_img[i * cell_size_y:(i + 1) * cell_size_y, j * cell_size_x:(j + 1) * cell_size_x])
            cell_direction = ((np.mean(ang_img[i * cell_size_y:(i + 1) * cell_size_y, j * cell_size_x:(j + 1) * cell_size_x]) + 180) / 360.0)

            vector_field[i, j, 0] = (cell_magnitude / 255.0) * np.cos(np.deg2rad(cell_direction))
            vector_field[i, j, 1] = (cell_magnitude / 255.0) * np.sin(np.deg2rad(cell_direction))

            subsampled_image[i, j] = cell_magnitude

    # normalize the images
    subsampled_image, vector_field = subsampled_image / 255.0, vector_field
    
    # add padding back at 4 sides when the grid size is not a divisor of the image size and is too large
    subsampled_image = np.pad(subsampled_image, ((0, 0), (0, 1)), mode='edge')
    vector_field = np.pad(vector_field, ((0, 0), (0, 1), (0, 0)), mode='edge')
    subsampled_image = np.pad(subsampled_image, ((0, 1), (0, 0)), mode='edge')
    vector_field = np.pad(vector_field, ((0, 1), (0, 0), (0, 0)), mode='edge')
    
    return subsampled_image, vector_field

# Set layout
st.set_page_config(layout="wide")

# Set title and instructions
st.title("Interactive Image Gradient Estimation")
st.write("## Upload an image to compute its gradient.")

initial_img = get_initial_image_as_bytes()

# Set image uploader
uploaded_img = st.sidebar.file_uploader("Upload an image (jpg, png)", type=["jpg", "png"])

# Add a slider in the sidebar for the kernel size
sobel_kernel_size = st.sidebar.slider('Sobel Kernel Size', 1, 7, step=2, value=3)

# resolution = st.sidebar.slider('Vector Field Resolution', 10, 100, step=10, value=50)

# Initial Display
# col1, col2, col3, col4 = st.columns(4)
col1, col2, col3 = st.columns(3)
col1.header("Original Image")
col2.header("Image Gradient Magnitude")
col3.header("Image Gradient Direction")
# col4.header("Subsampled Image")

img_to_use = initial_img if uploaded_img is None else Image.open(uploaded_img)

# Convert image to openCV color space (BGR to RGB)
img_as_array = np.array(img_to_use)

# Gradient computation
mag_img, ang_img = get_gradient_image(img_as_array, sobel_kernel_size)

# # Subsampling
# subsampled_img, vector_field = grid_based_subsampling(mag_img, ang_img, resolution)

mag_img = Image.fromarray(mag_img)
ang_img = Image.fromarray(ang_img)


col1.image(img_to_use, use_column_width=True)
col2.image(mag_img, use_column_width=True)
col3.image(ang_img, use_column_width=True)
# col4.image(subsampled_img, use_column_width=True)

# Calculate the base64 strings for the images
orig_img_bytes = pil_to_bytes(img_to_use)
b64_orig = bytes_to_base64(orig_img_bytes)

processed_img_mag_bytes = pil_to_bytes(mag_img)
b64_processed_mag = bytes_to_base64(processed_img_mag_bytes)

processed_img_ang_bytes = pil_to_bytes(ang_img)
b64_processed_ang = bytes_to_base64(processed_img_ang_bytes)

# Create download links
st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_orig}" download="original.png" class="download-button">Download Original Image</a>',
    unsafe_allow_html=True)

st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_processed_mag}" download="processed_magnitude.png" class="download-button">Download Gradient Magnitude Image</a>',
    unsafe_allow_html=True)

st.sidebar.markdown(
    f'<a href="data:image/png;base64,{b64_processed_ang}" download="processed_angle.png" class="download-button">Download Gradient Direction Image</a>',
    unsafe_allow_html=True)

# Inject CSS to make the hyperlink looks like a button from Streamlit
st.markdown(common_markdown(), unsafe_allow_html=True)

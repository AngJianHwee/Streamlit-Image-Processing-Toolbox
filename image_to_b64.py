import streamlit as st
from PIL import Image
import io

from _utils import pil_to_bytes, bytes_to_base64, get_initial_image_as_bytes, common_markdown

def compress_image(image_pil, quality):
    buffer = io.BytesIO()
    image_pil.save(buffer, format='JPEG', quality=quality)
    compressed = Image.open(buffer)
    
    return compressed, len(buffer.getvalue())

# Set layout information
st.set_page_config(layout="wide")

# Set title and instructions
st.title("Interactive Image to Base64 Conversion")
st.write("## Upload an image for compression and conversion to Base64.")

initial_img = get_initial_image_as_bytes()

# Set image uploader and compression slider
uploaded_img = st.sidebar.file_uploader("Upload an image (jpg, png)", type=["jpg", "png"])
compression_quality = st.sidebar.slider("Compression Quality (100 - little/none compression, 0 - high compression)", 0, 100, 100, 5)

# Displaying images
col1, col2, col3 = st.columns(3)
col1.header("Original Image")
col2.header("Compressed Image")

# Use the uploaded image, if present, otherwise use the initial image
img_to_use = initial_img if uploaded_img is None else Image.open(uploaded_img)

img_compressed, img_compressed_size = compress_image(img_to_use, compression_quality)

col1.image(img_to_use, use_column_width=True)
col2.image(img_compressed, use_column_width=True)

# Calculate the base64 strings for the images
orig_img_bytes = pil_to_bytes(img_to_use)
b64_orig = bytes_to_base64(orig_img_bytes)

compressed_img_bytes = pil_to_bytes(img_compressed)
b64_compressed = bytes_to_base64(compressed_img_bytes)


# Display the Base64 strings
col3.header("Base64")

col3.write(f"Original Image")
col3.write(f"-> Size (KB): {len(orig_img_bytes) / 1024:.2f}")
col3.write(f"-> Length of Base64 string: {len(b64_orig):,}")
col3.text_area("", value=b64_orig, height=250, max_chars=None)

col3.write(f"Compressed Image")
col3.write(f"-> Size (KB): {img_compressed_size / 1024:.2f}")
col3.write(f"-> Length of Base64 string: {len(b64_compressed):,}")
col3.text_area("", value=b64_compressed, height=250, max_chars=None)

# Create download link for the Base64 text (as a txt file)
st.sidebar.markdown(
    f'<a href="data:text/plain;charset=utf-8;base64,{b64_orig}" download="orig_image_b64.txt" class="download-button">Download Original Base64 txt file</a>',
    unsafe_allow_html=True)

st.sidebar.markdown(
    f'<a href="data:text/plain;charset=utf-8;base64,{b64_compressed}" download="compressed_image_b64.txt" class="download-button">Download Compressed Base64 txt file</a>',
    unsafe_allow_html=True)

# Inject CSS to make the hyperlink looks like a button from Streamlit
st.markdown(common_markdown(), unsafe_allow_html=True)

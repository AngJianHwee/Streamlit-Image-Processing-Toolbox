import io
from PIL import Image
import base64


# Function to convert PIL image to bytes
def pil_to_bytes(image: Image) -> bytes:
    byteIO = io.BytesIO()
    image.save(byteIO, format='PNG')
    return byteIO.getvalue()


# Function to convert bytes to base64
def bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


def get_initial_image_as_bytes(type="anime_sushi") -> bytes:
    if type == "anime_sushi":
        with open("./data/anime_sushi_b64.txt", "r") as f:
            base64_str = f.read()
    elif type == "anime_face":
        with open("./data/anime_face_b64.txt", "r") as f:
            base64_str = f.read()
    elif type == "human_face":
        with open("./data/human_face_b64.txt", "r") as f:
            base64_str = f.read()
    else:
        raise ValueError("Invalid type")
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))


def common_markdown():
    common_markdown = """
    <style>
    .download-button {
        display: inline-block;
        margin: 1em 0;
        padding: 0.5em 1em;
        border: 2px solid rgb(83, 84, 92);
        border-radius: 8px;
        font-size: 1em;
        text-align: center;
        color: rgb(216, 250, 237);
        background-color: rgb(43, 44, 54);
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        text-decoration: none;
    }
    .download-button:hover {
        color: rgb(221, 75, 75);
        border: 2px solid rgb(221, 75, 75);
        text-decoration: none;
    }
    </style>
    """

    return common_markdown
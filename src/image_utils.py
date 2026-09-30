import cv2
import numpy as np
from PIL import Image

def load_image(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode the uploaded image.")
    return bgr

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def image_info(image):
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "width": w, "height": h, "channels": channels,
        "dtype": str(image.dtype), "min": int(image.min()),
        "max": int(image.max()), "shape": str(image.shape)
    }

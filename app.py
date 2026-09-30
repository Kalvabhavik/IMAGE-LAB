import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.image_utils import load_image, bgr_to_rgb, image_info
from src.metrics import mse, psnr
from src.processing import (
    resize_image, quantize, enhance, restore,
    segment, morphology, edges, contours, checkerboard
)

st.set_page_config(page_title="ImageLab", page_icon="", layout="wide")

st.title("ImageLab")
st.caption("Interactive Digital Image Processing Studio")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
    )
    camera = st.camera_input("Or capture from camera")
    st.divider()
    

source = camera or uploaded

if source is None:
    st.markdown("""
    ## Welcome to ImageLab 

    Upload an image or capture one with your camera to start.

    **Suggested experiment path**

    `Image → Sampling → Quantization → Enhancement → Restoration → Segmentation → Morphology → Detection`

    This follows the practical digital image-processing pipeline from the course notes.
    """)
    st.image(checkerboard(), caption="Sampling experiment", width=350)
    st.stop()

image = load_image(source)
info = image_info(image)

with st.expander("stre Image Information", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Width", info["width"])
    c2.metric("Height", info["height"])
    c3.metric("Channels", info["channels"])
    c4.metric("Data Type", info["dtype"])
    c5.metric("Range", f'{info["min"]}–{info["max"]}')

tabs = st.tabs([
    " Explorer", " Sampling", " Quantization",
    " Enhancement", " Restoration", " Segmentation",
    " Morphology", " Edges & Objects"
])

with tabs[0]:
    st.subheader("Image Explorer")
    col1, col2 = st.columns(2)
    with col1:
        st.image(bgr_to_rgb(image), caption="Original RGB view", use_container_width=True)
    with col2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(gray, caption="Grayscale", use_container_width=True)

    fig, ax = plt.subplots()
    ax.hist(gray.ravel(), bins=256)
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel Count")
    st.pyplot(fig)
    plt.close(fig)

    channels = cv2.split(image)
    cols = st.columns(3)
    for col, name, ch in zip(cols, ["Blue", "Green", "Red"], channels):
        col.image(ch, caption=f"{name} channel", use_container_width=True)

with tabs[1]:
    st.subheader(" Sampling & Resolution Lab")
    scale = st.slider("Scale", 0.1, 2.0, 0.5, 0.1)
    method = st.selectbox("Interpolation", ["Nearest", "Bilinear", "Bicubic", "Lanczos"])
    resized = resize_image(image, scale, method)

    c1, c2 = st.columns(2)
    c1.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
    c2.image(bgr_to_rgb(resized), caption=f"{method} × {scale}", use_container_width=True)

    st.markdown("### Aliasing demonstration")
    board = checkerboard()
    anti = cv2.resize(
        cv2.GaussianBlur(board, (5, 5), 0), (128, 128),
        interpolation=cv2.INTER_AREA
    )
    no_anti = cv2.resize(board, (128, 128), interpolation=cv2.INTER_NEAREST)
    c1, c2 = st.columns(2)
    c1.image(bgr_to_rgb(no_anti), caption="Without pre-filter", use_container_width=True)
    c2.image(bgr_to_rgb(anti), caption="Gaussian pre-filter", use_container_width=True)

with tabs[2]:
    st.subheader(" Quantization Lab")
    bits = st.select_slider("Bit depth", options=[1, 2, 4, 8], value=4)
    qimg = quantize(image, bits)
    c1, c2 = st.columns(2)
    c1.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
    c2.image(bgr_to_rgb(qimg), caption=f"{bits}-bit quantized", use_container_width=True)
    st.metric("MSE", f"{mse(image, qimg):.2f}")
    p = psnr(image, qimg)
    st.metric("PSNR", "∞" if np.isinf(p) else f"{p:.2f} dB")

with tabs[3]:
    st.subheader(" Enhancement Lab")
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", 0.2, 3.0, 1.0, 0.1)
    mode = st.selectbox(
        "Enhancement",
        ["None", "Histogram Equalization", "CLAHE", "Sharpen", "Gaussian Denoise"]
    )
    result = enhance(image, brightness, contrast, mode)
    c1, c2 = st.columns(2)
    c1.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
    if result.ndim == 2:
        c2.image(result, caption=mode, use_container_width=True)
    else:
        c2.image(bgr_to_rgb(result), caption=mode, use_container_width=True)

with tabs[4]:
    st.subheader(" Restoration Lab")
    mode = st.selectbox("Filter", ["Median", "Gaussian", "Bilateral"])
    kernel = st.slider("Kernel size", 3, 11, 5, 2)
    result = restore(image, mode, kernel)
    c1, c2 = st.columns(2)
    c1.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
    c2.image(bgr_to_rgb(result), caption=mode, use_container_width=True)

with tabs[5]:
    st.subheader(" Segmentation Lab")
    mode = st.selectbox("Segmentation method", ["Binary", "Otsu", "Adaptive"])
    threshold = st.slider("Threshold", 0, 255, 127)
    result = segment(image, mode, threshold)
    c1, c2 = st.columns(2)
    c1.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
    c2.image(result, caption=mode, use_container_width=True)

with tabs[6]:
    st.subheader(" Morphological Processing")
    seg = segment(image, "Otsu")
    operation = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing"])
    kernel = st.slider("Kernel size", 3, 15, 5, 2)
    result = morphology(seg, operation, kernel)
    c1, c2 = st.columns(2)
    c1.image(seg, caption="Binary input", use_container_width=True)
    c2.image(result, caption=operation, use_container_width=True)

with tabs[7]:
    st.subheader("🔎 Edge & Object Analysis")
    mode = st.selectbox("Analysis", ["Canny", "Sobel", "Contours + Bounding Boxes"])
    if mode in ["Canny", "Sobel"]:
        result = edges(image, mode)
        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
        c2.image(result, caption=mode, use_container_width=True)
    else:
        result, count = contours(image)
        st.metric("Detected contour regions", count)
        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
        c2.image(bgr_to_rgb(result), caption="Bounding boxes", use_container_width=True)

st.divider()
st.caption("ImageLab • Built for learning and experimenting with Digital Image Processing")

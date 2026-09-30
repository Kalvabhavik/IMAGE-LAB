import cv2
import numpy as np

def resize_image(image, scale, method):
    methods = {
        "Nearest": cv2.INTER_NEAREST,
        "Bilinear": cv2.INTER_LINEAR,
        "Bicubic": cv2.INTER_CUBIC,
        "Lanczos": cv2.INTER_LANCZOS4,
    }
    h, w = image.shape[:2]
    return cv2.resize(
        image, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=methods[method]
    )

def quantize(image, bits):
    levels = 2 ** bits
    step = 256 / levels
    q = np.floor(image.astype(np.float32) / step) * step + step / 2
    return np.clip(q, 0, 255).astype(np.uint8)

def enhance(image, brightness=0, contrast=1.0, mode="None"):
    result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    if mode == "Histogram Equalization":
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    if mode == "CLAHE":
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if mode == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(result, -1, kernel)

    if mode == "Gaussian Denoise":
        return cv2.GaussianBlur(result, (5, 5), 0)

    return result

def restore(image, mode, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    if mode == "Median":
        return cv2.medianBlur(image, kernel_size)
    if mode == "Gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    if mode == "Bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image

def segment(image, mode, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "Binary":
        _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif mode == "Otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        out = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    return out

def morphology(binary, operation, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    ops = {
        "Erosion": cv2.MORPH_ERODE,
        "Dilation": cv2.MORPH_DILATE,
        "Opening": cv2.MORPH_OPEN,
        "Closing": cv2.MORPH_CLOSE,
    }
    return cv2.morphologyEx(binary, ops[operation], kernel)

def edges(image, mode):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "Sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        return cv2.convertScaleAbs(magnitude)
    return cv2.Canny(gray, 100, 200)

def contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, f"area={int(area)}", (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        count += 1
    return output, count

def checkerboard(size=512, squares=16):
    board = np.indices((size, size)).sum(axis=0) // (size // squares)
    board = (board % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

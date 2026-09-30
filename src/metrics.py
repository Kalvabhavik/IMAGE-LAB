import numpy as np

def mse(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def psnr(a, b):
    value = mse(a, b)
    if value == 0:
        return float("inf")
    return float(10 * np.log10((255.0 ** 2) / value))

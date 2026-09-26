import cv2
import numpy as np


def gray32(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32, copy=False)


def normalize_map(a: np.ndarray) -> np.ndarray:
    return cv2.normalize(a, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

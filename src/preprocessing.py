import warnings

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects

# skimage 0.26 renamed remove_small_objects' `min_size` kwarg to
# `max_size` with near-identical semantics for our use; this only
# silences the resulting FutureWarning so pipeline logs stay readable.
warnings.filterwarnings(
    "ignore", message=".*min_size.*", category=FutureWarning
)

# Resize all images to a common size
IMAGE_SIZE = (512, 512)

BLOCK_SIZE = 16  # block size for orientation-field estimation


def load_image(path):
    """
    Load fingerprint image from disk.
    """
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    return image


def resize_image(image):
    """
    Resize image to a fixed size.
    """
    return cv2.resize(image, IMAGE_SIZE)


def to_grayscale(image):
    """
    Convert BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_image(image):
    """
    Normalize intensity values to improve consistency.
    """
    return cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )


def enhance_contrast(gray):
    """
    Improve local contrast using CLAHE.
    """
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def remove_noise(image):
    """
    Remove noise while preserving fingerprint ridges.

    A median filter alone is not enough for these images: the source
    scans carry a fine halftone/print-dot pattern that survives a 3x3
    median blur and gets mistaken for ridges by the binarizer. A
    slightly larger Gaussian blur smooths that dot pattern out while
    the (much lower-frequency) ridge/valley structure survives.
    """
    denoised = cv2.medianBlur(image, 3)
    denoised = cv2.GaussianBlur(denoised, (5, 5), 0)
    return denoised


def estimate_orientation_field(gray, block_size=BLOCK_SIZE):
    """
    Estimate the local ridge-orientation field using the classic
    gradient-based least-squares method (Hong, Wan & Jain, 1998).

    Returns a per-pixel orientation map (radians, mod pi) so any
    pixel/minutia location can look up the ridge direction at its
    position instead of relying on a noisy single-pixel gradient.
    """
    gray_f = gray.astype(np.float32)

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    h, w = gray.shape

    # accumulate gradient covariance per block
    vx = np.zeros((h // block_size + 1, w // block_size + 1), dtype=np.float32)
    vy = np.zeros_like(vx)

    for by, y in enumerate(range(0, h, block_size)):
        for bx, x in enumerate(range(0, w, block_size)):
            gx_block = gx[y:y + block_size, x:x + block_size]
            gy_block = gy[y:y + block_size, x:x + block_size]

            vx[by, bx] = np.sum(2 * gx_block * gy_block)
            vy[by, bx] = np.sum(gx_block ** 2 - gy_block ** 2)

    # smooth the vector field (average in sin/cos space) to reduce noise
    vx_smooth = cv2.GaussianBlur(vx, (5, 5), 0)
    vy_smooth = cv2.GaussianBlur(vy, (5, 5), 0)

    block_orientation = 0.5 * np.arctan2(vx_smooth, vy_smooth)

    # upsample block orientation back to full resolution
    orientation_field = cv2.resize(
        block_orientation,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return orientation_field


def segment_foreground(gray, block_size=BLOCK_SIZE, thresh_ratio=0.25):
    """
    Rough fingerprint foreground/background segmentation using
    block-wise intensity variance. Background (blank margin / print
    artifacts outside the finger area) has much lower local variance
    than genuine ridge texture.
    """
    h, w = gray.shape

    stds = np.zeros((h // block_size + 1, w // block_size + 1), dtype=np.float32)

    for by, y in enumerate(range(0, h, block_size)):
        for bx, x in enumerate(range(0, w, block_size)):
            patch = gray[y:y + block_size, x:x + block_size]
            stds[by, bx] = patch.std()

    threshold = stds.max() * thresh_ratio
    block_mask = (stds > threshold).astype(np.uint8)

    mask = cv2.resize(
        block_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    return mask.astype(bool)


def threshold_image(image):
    """
    Convert grayscale image into binary image.
    Fingerprint ridges become black.

    Otsu's global threshold on the denoised image separates ridge
    from valley far more cleanly here than a small-window adaptive
    threshold, which was locking onto the halftone/print-dot noise
    instead of the ridge pattern.
    """

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def clean_image(binary, foreground_mask=None):
    """
    Remove small noise using morphological operations and
    connected-component filtering.
    """

    kernel = np.ones((5, 5), np.uint8)

    # Fill tiny gaps in ridges
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Remove isolated white pixels
    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    ridge_mask = (opened == 0)

    if foreground_mask is not None:
        ridge_mask = ridge_mask & foreground_mask

    # drop small speckle blobs on both the ridge side and the valley
    # side -- this is what actually removes the leftover halftone
    # noise that survives thresholding
    ridge_mask = remove_small_objects(ridge_mask, min_size=150)
    valley_mask = remove_small_objects(~ridge_mask, min_size=150)
    ridge_mask = ~valley_mask

    if foreground_mask is not None:
        ridge_mask = ridge_mask & foreground_mask

    cleaned = np.where(ridge_mask, 0, 255).astype(np.uint8)

    return cleaned


def prune_spurs(skeleton, iterations=6):
    """
    Iteratively remove skeleton endpoints to eliminate the short
    spurious spurs (1-6 px) that a raw skeletonize() call leaves
    behind. These spurs are the single biggest source of false
    ridge-endings in minutiae extraction.
    """
    skel = skeleton.copy().astype(np.uint8)
    neighbor_kernel = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=np.uint8
    )

    for _ in range(iterations):
        neighbor_count = cv2.filter2D(
            skel, -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT
        )
        endpoints = (skel == 1) & (neighbor_count == 1)
        skel[endpoints] = 0

    return skel


def skeletonize_image(binary):
    """
    Convert fingerprint ridges to one-pixel-wide skeleton, then
    prune spurious spurs left behind by the thinning algorithm.
    """

    # Fingerprint ridges should be True
    ridge = (binary == 0)

    skeleton = skeletonize(ridge).astype(np.uint8)

    skeleton = prune_spurs(skeleton, iterations=6)

    return skeleton


def preprocess(path):
    """
    Complete preprocessing pipeline.

    Returns a dictionary containing all intermediate images.
    """

    original = load_image(path)

    resized = resize_image(original)

    gray = to_grayscale(resized)

    normalized = normalize_image(gray)

    enhanced = enhance_contrast(normalized)

    foreground_mask = segment_foreground(normalized)

    orientation_field = estimate_orientation_field(enhanced)

    denoised = remove_noise(enhanced)

    binary = threshold_image(denoised)

    cleaned = clean_image(binary, foreground_mask)

    skeleton = skeletonize_image(cleaned)

    return {
        "original": original,
        "resized": resized,
        "gray": gray,
        "normalized": normalized,
        "enhanced": enhanced,
        "foreground_mask": foreground_mask,
        "orientation_field": orientation_field,
        "denoised": denoised,
        "binary": binary,
        "cleaned": cleaned,
        "skeleton": skeleton
    }

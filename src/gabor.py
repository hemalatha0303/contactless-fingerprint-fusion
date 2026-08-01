import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Configuration
# -----------------------------
KSIZE = 31
SIGMA = 4
LAMBDA = 8
GAMMA = 0.5
PSI = 0

NUM_ORIENTATIONS = 8
BLOCK_SIZE = 32


# -----------------------------
# Build Gabor Filter Bank
# -----------------------------
def build_gabor_bank():
    """
    Create Gabor filters at multiple orientations.
    """

    filters = []

    for theta in np.linspace(
        0,
        np.pi,
        NUM_ORIENTATIONS,
        endpoint=False
    ):

        kernel = cv2.getGaborKernel(
            (KSIZE, KSIZE),
            SIGMA,
            theta,
            LAMBDA,
            GAMMA,
            PSI,
            ktype=cv2.CV_32F
        )

        kernel /= 1.5 * kernel.sum()

        filters.append(kernel)

    return filters


# -----------------------------
# Apply Filters
# -----------------------------
def apply_gabor(image, filters):
    """
    Apply each Gabor filter.
    """

    responses = []

    for kernel in filters:

        response = cv2.filter2D(
            image,
            cv2.CV_32F,
            kernel
        )

        responses.append(response)

    return responses


# -----------------------------
# Extract Block Features
# -----------------------------
def block_features(response):
    """
    Extract local statistics from image blocks.
    """

    features = []

    h, w = response.shape

    for y in range(0, h, BLOCK_SIZE):

        for x in range(0, w, BLOCK_SIZE):

            block = response[
                y:y + BLOCK_SIZE,
                x:x + BLOCK_SIZE
            ]

            if block.size == 0:
                continue

            mean = np.mean(block)
            std = np.std(block)

            energy = np.sum(block ** 2)

            features.extend([
                mean,
                std,
                energy
            ])

    return features


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(responses):
    """
    Build complete texture descriptor.
    """

    feature_vector = []

    for response in responses:

        feature_vector.extend(
            block_features(response)
        )

    return np.array(
        feature_vector,
        dtype=np.float32
    )


# -----------------------------
# Similarity
# -----------------------------
def texture_similarity(vec1, vec2):
    """
    Cosine similarity between feature vectors.
    """

    score = cosine_similarity(
        vec1.reshape(1, -1),
        vec2.reshape(1, -1)
    )

    return float(score[0][0])


# -----------------------------
# Complete Pipeline
# -----------------------------

GABOR_FILTERS = build_gabor_bank()
def extract_texture_features(image):
    """
    Compute Gabor texture descriptor.
    """

    #filters = build_gabor_bank()

    responses = apply_gabor(image, GABOR_FILTERS)
    return extract_features(responses)

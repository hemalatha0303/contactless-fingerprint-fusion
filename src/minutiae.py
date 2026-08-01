import cv2
import numpy as np

# Ignore minutiae near image borders
BORDER_MARGIN = 20

# Minimum distance between two minutiae
MIN_DISTANCE = 10


def get_neighbors(img, x, y):
    """
    Return 8-neighbors in clockwise order.
    """
    return [
        img[y - 1, x],
        img[y - 1, x + 1],
        img[y, x + 1],
        img[y + 1, x + 1],
        img[y + 1, x],
        img[y + 1, x - 1],
        img[y, x - 1],
        img[y - 1, x - 1]
    ]


def crossing_number(neighbors):
    """
    Compute Crossing Number (CN).
    CN = 1 → Ridge Ending
    CN = 3 → Bifurcation
    """

    cn = 0

    for i in range(8):
        cn += abs(int(neighbors[i]) - int(neighbors[(i + 1) % 8]))

    return cn / 2


def estimate_orientation(orientation_field, x, y):
    """
    Look up the local ridge orientation at (x, y) from a precomputed
    orientation field (block-based least-squares gradient estimate),
    instead of differencing raw 0/1 skeleton pixel values -- the
    latter is degenerate on a 1-pixel-wide line and gives a very
    noisy direction estimate.
    """

    angle_rad = orientation_field[y, x]

    return np.degrees(angle_rad) % 180


def remove_duplicates(minutiae):
    """
    Remove minutiae detected very close to each other.
    """

    filtered = []

    for m in minutiae:

        keep = True

        for f in filtered:

            distance = np.sqrt(
                (m["x"] - f["x"]) ** 2 +
                (m["y"] - f["y"]) ** 2
            )

            if distance < MIN_DISTANCE:
                keep = False
                break

        if keep:
            filtered.append(m)

    return filtered


def extract_minutiae(skeleton, orientation_field=None):
    """
    Detect ridge endings and bifurcations.

    orientation_field: per-pixel ridge orientation map (radians) as
    produced by preprocessing.estimate_orientation_field. If not
    supplied, falls back to a flat 0-degree orientation for every
    minutia (keeps backward compatibility, e.g. for test_pipeline.py).
    """

    minutiae = []

    height, width = skeleton.shape

    if orientation_field is None:
        orientation_field = np.zeros((height, width), dtype=np.float32)

    for y in range(BORDER_MARGIN, height - BORDER_MARGIN):

        for x in range(BORDER_MARGIN, width - BORDER_MARGIN):

            if skeleton[y, x] != 1:
                continue

            neighbors = get_neighbors(skeleton, x, y)

            cn = crossing_number(neighbors)

            if cn not in [1, 3]:
                continue

            orientation = estimate_orientation(
                orientation_field,
                x,
                y
            )

            minutiae.append({

                "x": x,

                "y": y,

                "type": "ending" if cn == 1 else "bifurcation",

                "orientation": orientation

            })

    minutiae = remove_duplicates(minutiae)

    return minutiae


def draw_minutiae(image, minutiae):
    """
    Draw detected minutiae on fingerprint image.
    """

    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()

    for point in minutiae:

        if point["type"] == "ending":
            color = (0, 0, 255)      # Red
        else:
            color = (0, 255, 0)      # Green

        cv2.circle(
            output,
            (point["x"], point["y"]),
            3,
            color,
            -1
        )

        # Draw orientation line
        angle = np.radians(point["orientation"])

        x2 = int(point["x"] + 10 * np.cos(angle))
        y2 = int(point["y"] + 10 * np.sin(angle))

        cv2.line(
            output,
            (point["x"], point["y"]),
            (x2, y2),
            (255, 0, 0),
            1
        )

    return output

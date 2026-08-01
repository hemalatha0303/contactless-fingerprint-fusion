import math
import numpy as np
from scipy.spatial import cKDTree

# Maximum distance between two minutiae to be considered a candidate match
MAX_DISTANCE = 15

# Maximum orientation difference (degrees, mod 180) to be considered
# a candidate match
MAX_ORIENTATION_DIFF = 30

# Search grid used to rigidly align set1 onto set2 before matching.
# Two independent captures of the same finger are rarely placed at
# the exact same position/rotation, so matching raw (x, y) without
# first registering the two point clouds throws away most genuine
# correspondences.
TRANSLATIONS = [
    (dx, dy)
    for dx in range(-40, 41, 4)
    for dy in range(-40, 41, 4)
]
ROTATIONS_DEG = list(range(-12, 13, 4))


def _to_array(minutiae):
    if len(minutiae) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int8)

    xy = np.array([[m["x"], m["y"]] for m in minutiae], dtype=np.float32)
    orientation = np.array([m["orientation"] for m in minutiae], dtype=np.float32)
    # encode type as int for fast vectorized comparison
    types = np.array([0 if m["type"] == "ending" else 1 for m in minutiae], dtype=np.int8)

    return xy, orientation, types


def _orientation_diff(a, b):
    """
    Angular difference mod 180 degrees (ridge orientation is
    unsigned), wrapped into [0, 90].
    """
    diff = np.abs(a - b) % 180
    return np.minimum(diff, 180 - diff)


def _best_alignment(xy1, orient1, type1, xy2, orient2, type2, tree2):
    """
    Vectorized search over a grid of rigid transforms (rotation +
    translation). For every candidate transform, count how many
    points in the transformed set1 land within MAX_DISTANCE of a
    same-type, orientation-compatible point in set2 (nearest neighbor
    only -- not yet enforcing one-to-one uniqueness, which is a cheap
    and standard approximation for choosing the best alignment).

    Returns the (dx, dy, angle_deg) with the highest approximate
    match count.
    """
    n1 = len(xy1)
    center = xy1.mean(axis=0)

    best_score = -1
    best_transform = (0, 0, 0)

    for angle in ROTATIONS_DEG:
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        rotated = (xy1 - center) @ rot.T + center
        rotated_orient = (orient1 + angle) % 180

        # stack every translation for this rotation into one big
        # array so the KD-tree query happens in a single vectorized
        # call instead of one call per (dx, dy)
        offsets = np.array(TRANSLATIONS, dtype=np.float32)  # (T, 2)
        all_points = rotated[np.newaxis, :, :] + offsets[:, np.newaxis, :]  # (T, n1, 2)
        flat_points = all_points.reshape(-1, 2)  # (T*n1, 2)

        distances, indices = tree2.query(flat_points, k=1)

        distances = distances.reshape(len(TRANSLATIONS), n1)
        indices = indices.reshape(len(TRANSLATIONS), n1)

        matched_type = type2[indices]
        matched_orient = orient2[indices]

        compatible = (
            (distances <= MAX_DISTANCE)
            & (matched_type == type1[np.newaxis, :])
            & (_orientation_diff(rotated_orient[np.newaxis, :], matched_orient) <= MAX_ORIENTATION_DIFF)
        )

        counts = compatible.sum(axis=1)  # (T,)
        best_t_idx = int(np.argmax(counts))

        if counts[best_t_idx] > best_score:
            best_score = counts[best_t_idx]
            dx, dy = TRANSLATIONS[best_t_idx]
            best_transform = (dx, dy, angle)

    return best_transform


def _exact_match_count(xy1, orient1, type1, xy2, orient2, type2, tree2):
    """
    One-to-one greedy assignment (closest pairs first) at a fixed
    alignment -- used only once, at the best transform found by
    _best_alignment, to get a precise match count for scoring.
    """
    if len(xy1) == 0:
        return 0

    k = min(4, len(xy2))
    distances, indices = tree2.query(xy1, k=k)

    if k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    candidates = []
    for i in range(len(xy1)):
        for c in range(k):
            d = distances[i, c]
            j = indices[i, c]
            if d > MAX_DISTANCE:
                continue
            if type1[i] != type2[j]:
                continue
            if _orientation_diff(orient1[i], orient2[j]) > MAX_ORIENTATION_DIFF:
                continue
            candidates.append((d, i, j))

    candidates.sort(key=lambda c: c[0])

    used_i, used_j = set(), set()
    matches = 0

    for d, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matches += 1

    return matches


def align_and_match(set1, set2):
    """
    Rigidly align set1 onto set2 (searching over a grid of rotations
    and translations) and return the best one-to-one match count.
    """

    xy1, orient1, type1 = _to_array(set1)
    xy2, orient2, type2 = _to_array(set2)

    if len(xy1) == 0 or len(xy2) == 0:
        return 0

    tree2 = cKDTree(xy2)

    dx, dy, angle = _best_alignment(
        xy1, orient1, type1, xy2, orient2, type2, tree2
    )

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    center = xy1.mean(axis=0)

    aligned_xy = (xy1 - center) @ rot.T + center + np.array([dx, dy], dtype=np.float32)
    aligned_orient = (orient1 + angle) % 180

    return _exact_match_count(
        aligned_xy, aligned_orient, type1,
        xy2, orient2, type2, tree2
    )


def similarity_score(set1, set2):
    """
    Similarity score between two minutiae sets, using rigid alignment
    followed by orientation-aware nearest-neighbor matching.
    """

    if len(set1) == 0 or len(set2) == 0:
        return 0.0

    matches = align_and_match(set1, set2)

    score = matches / max(len(set1), len(set2))

    return round(score, 4)

import os
import random
import pandas as pd
from itertools import combinations

ENROLLED_DIR = "data/enrolled"
OUTPUT_CSV = "data/pair_list.csv"

random.seed(42)


def group_by_finger(files):
    """
    Groups images by finger ID.
    Example:
        101 -> [101_1.tif, ..., 101_8.tif]
    """

    groups = {}

    for file in files:

        finger_id = file.split("_")[0]

        groups.setdefault(finger_id, []).append(file)

    return groups


def generate_genuine_pairs(groups):

    pairs = []

    for finger_id, images in groups.items():

        images = sorted(images)

        for img1, img2 in combinations(images, 2):

            pairs.append([
                img1,
                img2,
                1
            ])

    return pairs


def generate_impostor_pairs(groups, target_count):

    pairs = []

    finger_ids = list(groups.keys())

    while len(pairs) < target_count:

        f1, f2 = random.sample(finger_ids, 2)

        img1 = random.choice(groups[f1])

        img2 = random.choice(groups[f2])

        pairs.append([
            img1,
            img2,
            0
        ])

    return pairs


def main():

    files = [
        f for f in os.listdir(ENROLLED_DIR)
        if f.endswith(".tif")
    ]

    files.sort()

    groups = group_by_finger(files)

    genuine = generate_genuine_pairs(groups)

    impostor = generate_impostor_pairs(
        groups,
        len(genuine)
    )

    all_pairs = genuine + impostor

    random.shuffle(all_pairs)

    df = pd.DataFrame(
        all_pairs,
        columns=[
            "enrolled",
            "query",
            "label"
        ]
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("Finger IDs :", len(groups))
    print("Genuine Pairs :", len(genuine))
    print("Impostor Pairs :", len(impostor))
    print("Total Pairs :", len(df))

    print("\nSaved to", OUTPUT_CSV)


if __name__ == "__main__":
    main()

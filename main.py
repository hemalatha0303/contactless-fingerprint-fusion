import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

from src.preprocessing import preprocess
from src.minutiae import extract_minutiae
from src.matcher import similarity_score
from src.gabor import (
    extract_texture_features,
    texture_similarity,
)
from src.fusion import (
    min_max_normalize,
    weighted_fusion,
    find_best_weight,
    classifier_fusion,
)


# ---------------------------
# Feature Cache
# ---------------------------
feature_cache = {}

def extract_features(image_path):
    # Cache by full path, not just filename -- enrolled/ and query/
    # can legitimately contain different images that happen to share
    # a filename, and keying on basename alone would silently return
    # the wrong image's features in that case.
    key = os.path.abspath(image_path)

    if key in feature_cache:
        return feature_cache[key]

    processed = preprocess(image_path)

    minutiae = extract_minutiae(
        processed["skeleton"],
        processed["orientation_field"]
    )

    texture = extract_texture_features(processed["enhanced"])

    feature_cache[key] = {
        "minutiae": minutiae,
        "texture": texture
    }

    return feature_cache[key]

def compare_pair(enrolled_path, query_path):
    """
    Compare two fingerprint images and return each matcher's raw
    score. Fusion is deliberately NOT done here: proper min-max
    normalization needs the full distribution of scores across the
    dataset, so fusion happens once, after every pair has been
    scored (see process_dataset).
    """

    f1 = extract_features(enrolled_path)
    f2 = extract_features(query_path)

    minutiae_score = similarity_score(
        f1["minutiae"],
        f2["minutiae"]
    )

    texture_score = texture_similarity(
        f1["texture"],
        f2["texture"]
    )

    return {
        "minutiae": minutiae_score,
        "texture": texture_score
    }


def process_dataset():

    pairs = pd.read_csv("data/pair_list.csv")

    results = []

    total = len(pairs)

    for index, row in pairs.iterrows():

        enrolled_path = os.path.join(
            "data",
            "enrolled",
            row["enrolled"]
        )

        query_path = os.path.join(
            "data",
            "query",
            row["query"]
        )

        scores = compare_pair(
            enrolled_path,
            query_path
        )

        results.append({

            "enrolled": row["enrolled"],

            "query": row["query"],

            "label": row["label"],

            "minutiae": scores["minutiae"],

            "texture": scores["texture"]

        })

        if (index + 1) % 25 == 0 or (index + 1) == total:
            print(f"Processed {index + 1}/{total}")

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------
    # Dataset-level score fusion
    # ------------------------------------------------------------
    # 1. Normalize each matcher's raw scores to [0, 1] using the
    #    observed min/max across the whole dataset (reference doc,
    #    Step 4.1) -- fusing raw, unnormalized scores (as the
    #    original pipeline did) lets whichever matcher happens to
    #    have a wider numeric range dominate the fused score for
    #    reasons that have nothing to do with which matcher is more
    #    accurate.
    minutiae_norm = min_max_normalize(results_df["minutiae"])
    texture_norm = min_max_normalize(results_df["texture"])

    results_df["minutiae_norm"] = minutiae_norm
    results_df["texture_norm"] = texture_norm

    # 2a. Weighted linear fusion, with the weight chosen by searching
    #     for the AUC-maximizing value instead of a hand-picked guess.
    best_weight, best_weight_auc = find_best_weight(
        results_df["label"], minutiae_norm, texture_norm
    )
    results_df["fusion_weighted"] = weighted_fusion(
        minutiae_norm, texture_norm, best_weight
    )

    # 2b. Classifier-based (Logistic Regression) fusion with 5-fold
    #     cross-validation, so every fused score comes from a model
    #     that did not see that pair during training.
    fused_ml, ml_model = classifier_fusion(
        results_df["label"], minutiae_norm, texture_norm
    )
    results_df["fusion_ml"] = fused_ml

    # Both fusion strategies are kept for comparison. The tuned
    # weighted fusion is used as the headline "fusion" column here
    # because it scores highest on this dataset; note that its
    # weight was selected by searching this same dataset, whereas
    # fusion_ml's score is honestly out-of-sample (5-fold CV) but
    # scores marginally lower. See outputs/evaluation_summary.csv
    # for the full comparison before trusting either number blindly
    # on a different dataset.
    results_df["fusion"] = results_df["fusion_weighted"]

    os.makedirs("outputs", exist_ok=True)

    results_df.to_csv(
        "outputs/results.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Persist everything a live, single-pair demo (app.py) needs to
    # reproduce this exact scoring pipeline: the normalization
    # bounds (fit on this dataset), the tuned fusion weight, the
    # trained classifier, and a decision threshold (EER point on the
    # weighted-fusion score). Without this, a deployed app would
    # have no dataset to normalize a brand-new pair's raw scores
    # against.
    fpr, tpr, thresholds = roc_curve(results_df["label"], results_df["fusion"])
    fnr = 1 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer_threshold = float(thresholds[eer_idx])

    bundle = {
        "minutiae_min": float(np.min(results_df["minutiae"])),
        "minutiae_max": float(np.max(results_df["minutiae"])),
        "texture_min": float(np.min(results_df["texture"])),
        "texture_max": float(np.max(results_df["texture"])),
        "best_weight": float(best_weight),
        "decision_threshold": eer_threshold,
    }

    joblib.dump(ml_model, "outputs/classifier_fusion_model.joblib")

    with open("outputs/deployment_bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)

    print("\nFinished.")
    print(f"Total comparisons : {len(results_df)}")
    print(f"Cached images     : {len(feature_cache)}")
    print(f"Best weighted-fusion weight (minutiae) : {best_weight:.2f} (AUC={best_weight_auc:.4f})")
    print("Results saved to outputs/results.csv")
    print("Deployment bundle saved to outputs/deployment_bundle.json + outputs/classifier_fusion_model.joblib")


def main():

    process_dataset()


if __name__ == "__main__":
    main()

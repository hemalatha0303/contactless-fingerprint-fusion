import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc


def min_max_normalize(scores, min_val=None, max_val=None):
    """
    Normalize scores to the range [0, 1].

    Two matcher scores (minutiae match ratio, texture cosine
    similarity) live on different scales and distributions and
    cannot be combined directly -- they must be normalized to a
    common range first (reference doc, Step 4.1). min_val/max_val can
    be supplied so a query score can be normalized against the
    enrolled dataset's observed range instead of its own.
    """

    scores = np.asarray(scores, dtype=np.float64)

    if min_val is None:
        min_val = np.min(scores)
    if max_val is None:
        max_val = np.max(scores)

    denom = (max_val - min_val) if max_val != min_val else 1e-8

    normalized = (scores - min_val) / denom

    return np.clip(normalized, 0.0, 1.0)


def weighted_fusion(minutiae_norm, texture_norm, minutiae_weight=0.5):
    """
    Weighted linear score fusion on already-normalized scores.

    Final Score = w * minutiae + (1 - w) * texture
    """

    minutiae_norm = np.asarray(minutiae_norm, dtype=np.float64)
    texture_norm = np.asarray(texture_norm, dtype=np.float64)

    return (
        minutiae_weight * minutiae_norm +
        (1 - minutiae_weight) * texture_norm
    )


def find_best_weight(labels, minutiae_norm, texture_norm, steps=101):
    """
    Grid-search the fusion weight that maximizes ROC-AUC on the given
    (labeled) score set.
    """

    labels = np.asarray(labels)
    best_weight, best_auc = 0.5, -1.0

    for w in np.linspace(0.0, 1.0, steps):
        fused = weighted_fusion(minutiae_norm, texture_norm, w)
        fpr, tpr, _ = roc_curve(labels, fused)
        score_auc = auc(fpr, tpr)

        if score_auc > best_auc:
            best_auc = score_auc
            best_weight = w

    return best_weight, best_auc


def classifier_fusion(labels, minutiae_norm, texture_norm, n_splits=5, random_state=42):
    """
    Machine-learning (classifier-based) score fusion, per reference
    doc Step 4.2: train a Logistic Regression on
    [minutiae_norm, texture_norm] -> genuine/impostor and use its
    predicted probability as the fused score.

    Uses stratified k-fold cross-validation so every fused score is
    produced by a model that never saw that pair during training --
    this avoids the optimistic bias of fitting and scoring on the
    same pairs.
    """

    X = np.column_stack([minutiae_norm, texture_norm])
    y = np.asarray(labels)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = LogisticRegression()

    fused_probs = cross_val_predict(
        model, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    # also fit on the full data so the trained model can be reused
    # at inference time on new, unlabeled pairs
    full_model = LogisticRegression().fit(X, y)

    return fused_probs, full_model

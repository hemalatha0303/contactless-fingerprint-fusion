import json
import tempfile
import os

import joblib
import numpy as np
import streamlit as st

from src.preprocessing import preprocess
from src.minutiae import extract_minutiae
from src.matcher import similarity_score
from src.gabor import extract_texture_features, texture_similarity
from src.fusion import min_max_normalize, weighted_fusion

st.set_page_config(page_title="Fingerprint Fusion Matcher", page_icon="🔒", layout="centered")

BUNDLE_PATH = "outputs/deployment_bundle.json"
MODEL_PATH = "outputs/classifier_fusion_model.joblib"


@st.cache_resource
def load_bundle():
    """
    Load the normalization bounds, tuned fusion weight, decision
    threshold, and trained classifier that main.py fit on the full
    560-pair dataset. A live demo comparing a brand-new pair has no
    dataset of its own to normalize against, so it reuses the bounds
    fit during batch evaluation instead.
    """
    if not (os.path.exists(BUNDLE_PATH) and os.path.exists(MODEL_PATH)):
        return None, None

    with open(BUNDLE_PATH) as f:
        bundle = json.load(f)

    model = joblib.load(MODEL_PATH)

    return bundle, model


def score_pair(path_a, path_b):
    p1 = preprocess(path_a)
    p2 = preprocess(path_b)

    m1 = extract_minutiae(p1["skeleton"], p1["orientation_field"])
    m2 = extract_minutiae(p2["skeleton"], p2["orientation_field"])

    t1 = extract_texture_features(p1["enhanced"])
    t2 = extract_texture_features(p2["enhanced"])

    minutiae_score = similarity_score(m1, m2)
    texture_score = texture_similarity(t1, t2)

    return minutiae_score, texture_score, len(m1), len(m2)


def save_upload(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


st.title("🔒 Contactless Fingerprint Fusion Matcher")
st.caption(
    "Upload two fingerprint images to compare. Scores are produced by the "
    "same minutiae + Gabor-texture fusion pipeline evaluated in "
    "ANALYSIS_REPORT.md (AUC 0.864 on a 560-pair benchmark)."
)

bundle, model = load_bundle()

if bundle is None:
    st.error(
        "No deployment bundle found. Run `python main.py` once (locally or "
        "as a build step) to generate outputs/deployment_bundle.json and "
        "outputs/classifier_fusion_model.joblib before using this app."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    enrolled_file = st.file_uploader("Enrolled image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if enrolled_file:
        st.image(enrolled_file, caption="Enrolled", use_container_width=True)

with col2:
    query_file = st.file_uploader("Query image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if query_file:
        st.image(query_file, caption="Query", use_container_width=True)

if st.button("Compare", type="primary", disabled=not (enrolled_file and query_file)):
    with st.spinner("Extracting minutiae + texture features and matching..."):
        path_a = save_upload(enrolled_file)
        path_b = save_upload(query_file)

        try:
            minutiae_raw, texture_raw, n1, n2 = score_pair(path_a, path_b)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

        # normalize against the fitted dataset bounds, clipping to [0, 1]
        # in case a new pair's raw score falls outside the observed
        # training range
        minutiae_norm = min_max_normalize(
            [minutiae_raw], bundle["minutiae_min"], bundle["minutiae_max"]
        )[0]
        texture_norm = min_max_normalize(
            [texture_raw], bundle["texture_min"], bundle["texture_max"]
        )[0]

        fused_weighted = weighted_fusion(minutiae_norm, texture_norm, bundle["best_weight"])
        fused_ml = model.predict_proba([[minutiae_norm, texture_norm]])[0, 1]

        verdict = "MATCH (likely genuine)" if fused_weighted >= bundle["decision_threshold"] else "NO MATCH (likely impostor)"

    st.subheader("Result")
    if fused_weighted >= bundle["decision_threshold"]:
        st.success(verdict)
    else:
        st.warning(verdict)

    m1, m2, m3 = st.columns(3)
    m1.metric("Minutiae score", f"{minutiae_raw:.3f}", help=f"{n1} vs {n2} minutiae detected")
    m2.metric("Texture score", f"{texture_raw:.3f}")
    m3.metric("Fused score", f"{fused_weighted:.3f}", help=f"decision threshold: {bundle['decision_threshold']:.3f}")

    with st.expander("Details"):
        st.write(f"Normalized minutiae score: {minutiae_norm:.3f}")
        st.write(f"Normalized texture score: {texture_norm:.3f}")
        st.write(f"Weighted fusion (w={bundle['best_weight']:.2f} on minutiae): {fused_weighted:.3f}")
        st.write(f"Classifier-fusion probability (5-fold CV model): {fused_ml:.3f}")
        st.caption(
            "Normalization bounds and fusion weight were fit on a 560-pair "
            "benchmark (10 subjects), not on this uploaded pair -- see "
            "ANALYSIS_REPORT.md for the full evaluation and its limitations."
        )

st.divider()
if os.path.exists("outputs/roc_curve.png"):
    with st.expander("Benchmark ROC curve (from outputs/roc_curve.png)"):
        st.image("outputs/roc_curve.png")

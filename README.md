# Contactless Fingerprint Matching & Fusion

Advanced biometric matching pipeline: minutiae matcher + Gabor texture
matcher, fused via a tuned weighted sum (with a cross-validated
classifier fusion reported alongside it for comparison).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run

```bash
python main.py            # scores every pair in data/pair_list.csv -> outputs/results.csv
python src/evaluation.py  # ROC comparison + metrics -> outputs/roc_curve.png, outputs/evaluation_summary.csv
```

## Pipeline

```
Fingerprint Image
       │
       ▼
Image Preprocessing
   • Resize
   • Grayscale Conversion
   • CLAHE Contrast Enhancement
   • Gaussian Denoising
   • Adaptive/Otsu Thresholding
   • Morphological Cleaning
   • Skeletonization
       │
       ├──────────────────────────────┐
       ▼                              ▼
Minutiae Extraction            Gabor Texture Extraction
(Crossing Number)              (Multi-orientation Filters)
       │                              │
       ▼                              ▼
Minutiae Matching              Texture Similarity
       └──────────────┬───────────────┘
                      ▼
             Score Normalization
                      ▼
             Weighted Score Fusion
                      ▼
          Match / Non-Match Decision
                      ▼
      Performance Evaluation
 (ROC, AUC, FAR, FRR, TAR, EER)
```

See `ANALYSIS_REPORT.md` for the before/after comparison and why each
change was made.

## Project structure

```
data/                enrolled/, query/ images + pair_list.csv (genuine/impostor pairs)
src/
  preprocessing.py    denoising, binarization, skeletonization, orientation field
  minutiae.py          crossing-number minutiae extraction
  matcher.py            rigid alignment + orientation-aware minutiae matching
  gabor.py                Gabor filter-bank texture descriptor + cosine similarity
  fusion.py                score normalization, weighted fusion, classifier fusion
  evaluation.py             ROC/AUC/EER/TAR comparison across all systems
main.py                pipeline entry point
outputs/               results.csv, roc_curve.png, evaluation_summary.csv
```

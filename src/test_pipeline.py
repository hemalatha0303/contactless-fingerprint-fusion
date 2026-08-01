import cv2

from src.preprocessing import preprocess
from src.minutiae import extract_minutiae, draw_minutiae

# NOTE: this script opens GUI windows (cv2.imshow) and needs a
# display -- it's a manual/local debugging aid, not part of the
# automated main.py pipeline.
results = preprocess("data/enrolled/101_1.tif")

for name, image in results.items():
    if name == "orientation_field":
        continue  # not directly viewable as an image
    cv2.imshow(name, image)

skeleton = results["skeleton"]

minutiae = extract_minutiae(skeleton, results["orientation_field"])

print("Total Minutiae:", len(minutiae))

image = draw_minutiae(skeleton * 255, minutiae)

cv2.imshow("Minutiae", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# GAN for stylized/geometric image synthesis - CONTOUR/PALETTE EXTRACTION
# Dallas Scott (ds4015)

import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


INPUT_DIR   = '../datasets/train_B'
OUTPUT_DIR  = '../datasets/train_A_new'
K           = 8
SWATCH_H    = 64
CANNY_LOW   = 100
CANNY_HIGH  = 200
MAX_PIXELS  = 200_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith(('.png','.jpg','.jpeg')):
        continue

    # open image and convert to grayscale
    img_path = os.path.join(INPUT_DIR, fname)
    bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


    # edge detection - geometric/stylized
    #blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    #edges = cv2.Canny(blur, threshold1=50, threshold2=150)
    #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # detect contours with Canny
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    sketch = (255 - edges)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    H, W = sketch.shape

    # get colors from original image
    orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pixels = orig.reshape(-1,3).astype(np.float32)
    if len(pixels) > MAX_PIXELS:
        idx = np.random.choice(len(pixels), MAX_PIXELS, replace=False)
        pixels = pixels[idx]

    # extract palette colors
    km = KMeans(n_clusters=K, random_state=0).fit(pixels)
    colors = km.cluster_centers_.astype(np.uint8)

    # combine palette colors into a horizontal bar
    sw_w = W // K
    bar = Image.new('RGB', (W, SWATCH_H), (255,255,255))
    for i, c in enumerate(colors):
        swatch = Image.new('RGB', (sw_w, SWATCH_H), tuple(int(x) for x in c))
        bar.paste(swatch, (i*sw_w, 0))

    # combine sketch and palette bar into a single image
    top = Image.fromarray(sketch_rgb)
    combo = Image.new('RGB', (W, H + SWATCH_H), (255,255,255))
    combo.paste(top, (0, 0))
    combo.paste(bar, (0, H))

    combo.save(os.path.join(OUTPUT_DIR, fname))

print('Palette+contour images saved to', OUTPUT_DIR)
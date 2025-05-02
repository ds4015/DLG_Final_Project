import os
import json
from PIL import Image

# path to your annotation file and image folder
ann_path = "../datasets/instances_val2017.json"
img_folder = "../datasets/coco_stylized"

with open(ann_path, 'r') as f:
    coco = json.load(f)

for img in coco["images"]:
    img_file = os.path.join(img_folder, img["file_name"])
    try:
        with Image.open(img_file) as pil:
            w, h = pil.size
        if img["width"] != w or img["height"] != h:
            print(f"Fixing {img['file_name']}: {img['width']}×{img['height']} → {w}×{h}")
            img["width"], img["height"] = w, h
    except FileNotFoundError:
        print("Missing file:", img_file)

# write out a corrected annotations file
out_path = ann_path.replace(".json", "_fixed.json")
with open(out_path, 'w') as f:
    json.dump(coco, f)
print("Wrote corrected annotations to", out_path)

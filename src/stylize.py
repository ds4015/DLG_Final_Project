# COCO Common Objects Dataset Stylization
# Using val2017 dataset (5000 images)

import os
import torch
import random
import subprocess

# input dir of COCO images, output save, trained models path
input_dir = "C:/Users/ds/Downloads/val2017/val2017"
output_dir = "C:/Users/ds/Documents/GitHub/DLG_Final_Project/datasets/coco_stylized"
model_dir = "C:/Users/ds/Downloads/saved_models/saved_models"

# styles that will be randomly chosen
styles = [
    "mosaic.pth",
    "candy.pth",
    "rain_princess.pth",
    "udnie.pth"
]

os.makedirs(output_dir, exist_ok=True)

# stylize all coco images w/random style
for file in os.listdir(input_dir):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        input_path = os.path.join(input_dir, file)
        
        model_file = random.choice(styles)
        model_path = os.path.join(model_dir, model_file)
        style_name = os.path.splitext(model_file)[0]
        
        base_name, ext = os.path.splitext(file)
        output_file = f"{base_name}_{style_name}{ext}"
        output_path = os.path.join(output_dir, output_file)

        command = [
            "python", "neural_style.py", "eval",
            "--content-image", input_path,
            "--output-image", output_path,
            "--model", model_path,
            "--cuda", "0"
        ]
        
        subprocess.run(command)

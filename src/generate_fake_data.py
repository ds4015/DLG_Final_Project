

import json
import random

# 40 random objects
object_types = [
    "person", "car", "tree", "dog", "cat", "bike", "chair", "stool", "table", "lamp", "book",
    "phone", "cup", "hat", "shoe", "clock", "mirror", "window", "door", "bed", "bench",
    "computer", "rug", "television", "shelf", "plant", "bag", "glasses", "painting", "scarf",
    "umbrella", "towel", "candle", "fork", "spoon", "knife", "scissors", "pillow", "toothbrush", "fan"
]

image_data = {
    f"image_{i}": random.sample(object_types, k=random.randint(3, 5))
    for i in range(500)
}

# Save as JSON
with open("../dummy_object_dataset.json", "w") as f:
    json.dump(image_data, f, indent=2)

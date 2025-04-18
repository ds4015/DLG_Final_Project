import json
import random

with open("../dummy_object_dataset.json", "r") as f:
    dataset = json.load(f)

def find_images_with_object(obj_name):
    return [img_id for img_id, objects in dataset.items() if obj_name in objects]

if __name__ == "__main__":
    object = input("Enter object query: ")
    max_len = input("Enter max number of results (leave empty for all results): ")

    all_results = find_images_with_object(object)

    if len(all_results) == 0:
        print(f"'{object}' does not exist in dataset")

    if max_len == '':
        max_len = len(all_results)

    results = random.sample(all_results, int(max_len))
    print(f"Found {len(results)} images containing '{object}':")
    for img in results:
        print(img)

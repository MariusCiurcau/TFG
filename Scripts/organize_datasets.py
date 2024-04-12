import os
import shutil

import cv2
import numpy as np
import unicodedata
from PIL import Image


def normalize_string(string):
    # Replace accents with corresponding base characters
    normalized_string = ''.join(c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn')
    # Replace spaces with underscores
    normalized_string = normalized_string.replace(" ", "_")
    return normalized_string
def organize_images_and_labels(image_dir, label_dir, output_dir):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Iterate through image files
    i = 0
    last_dir = os.path.basename(os.path.normpath(output_dir))
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                image_name, _ = os.path.splitext(os.path.basename(file))
                image_name_normalized = normalize_string(image_name)

                image_path = os.path.join(root, file)
                label_path = os.path.join(label_dir, image_name_normalized + '.txt')

                # resize image (for Roboflow images)
                pil = Image.open(image_path).convert('RGB')
                image = np.array(pil)
                image = cv2.resize(image, (224, 224))

                image_destination_path = os.path.join(output_image_dir, f"{last_dir}_{i:04d}.jpg")
                label_destination_path = os.path.join(output_label_dir, f"{last_dir}_{i:04d}.txt")

                if os.path.exists(label_path):
                    cv2.imwrite(image_destination_path, image)
                    shutil.copy(label_path, label_destination_path)
                else:
                    print(f"Label file for {file} not found.")
            i += 1

def merge_dirs(input_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for source_dir in input_dirs:
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            destination_item = os.path.join(output_dir, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item)
            else:
                shutil.copy2(source_item, destination_item)

if __name__ == "__main__":
    images_dirs = ["../Datasets/Dataset/Femurs/Versiones anteriores/grayscale_images_old", "../Datasets/original_AO/resized_images",
                   ["../Datasets/FXMalaga/resized_images", "../Datasets/FracturasAQ/Data/resized_images"]]
    labels_dir = "../Datasets/Dataset/Femurs/labels/3clases/labels_fractura"
    output_dirs = ["../Datasets/ROB", "../Datasets/AO", "../Datasets/HVV"]

    for images_dir, output_dir in zip(images_dirs, output_dirs):
        if isinstance(images_dir, list):
            temp_dir = output_dir + "_temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            merge_dirs(images_dir, temp_dir)
            organize_images_and_labels(temp_dir, labels_dir, output_dir)
            shutil.rmtree(temp_dir)
        else:
            organize_images_and_labels(images_dir, labels_dir, output_dir)
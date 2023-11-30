import os
import cv2
import albumentations as A
from albumentations import (
    HorizontalFlip, RandomRotate90, VerticalFlip,
    RandomBrightnessContrast, ShiftScaleRotate,
    Blur, RandomGamma, Normalize, Rotate
)
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def augment_images(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder):
    # Define augmentation pipeline
    transform = A.Compose([
        #HorizontalFlip(p=0.5),
        #VerticalFlip(p=0.5),
        Rotate(p=0.5, limit=15),
        #RandomBrightnessContrast(p=0.2),
        #ShiftScaleRotate(p=0.2, rotate_limit=5),
        Blur(p=0.1),
        RandomGamma(p=0.1)
        #Normalize(),
        #ToTensorV2()
    ])

    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    # Augment and save images
    for image_file in os.listdir(input_images_folder):
        input_image_path = os.path.join(input_images_folder, image_file)
        input_label_path = os.path.join(input_labels_folder, os.path.splitext(image_file)[0] + '.txt')
        label = None
        if os.path.exists(input_label_path):
            with open(input_label_path, 'r') as file:
                label = int((file.readlines()[0]).split()[0])
        else:
            print(f"Image: {image_file}, label file not found.")

        original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

        image = original_image[:, :, None]  # Add a channel dimension to make it 3D
        image = image.astype('uint8')

        output_image_path = os.path.join(output_images_folder, f'{os.path.splitext(image_file)[0]}_0.jpg')
        cv2.imwrite(output_image_path, original_image)
        file = open(os.path.join(output_labels_folder, "{0}_{1}.txt".format(os.path.splitext(image_file)[0], 0)), 'w+')
        file.write(str(label))
        file.close()

        n_augmentations = 8 if label == 1 else 1

        for i in range(1, n_augmentations + 1):
            transformed = transform(image=image)
            transformed_image = transformed["image"].squeeze(axis=-1)
            output_image_path = os.path.join(output_images_folder, "{0}_{1}.jpg".format(os.path.splitext(image_file)[0],i))
            cv2.imwrite(output_image_path, transformed_image)
            file = open(os.path.join(output_labels_folder, "{0}_{1}.txt".format(os.path.splitext(image_file)[0],i)), 'w+')
            file.write(str(label))
            file.close()


        # Display original and augmented images
        #plt.figure(figsize=(8, 4))
        #plt.subplot(1, 2, 1)
        #plt.title("Original Image")
        #plt.imshow(original_image, cmap='gray')

        #plt.subplot(1, 2, 2)
        #plt.title("Augmented Image")
        #plt.imshow(image, cmap='gray')

        #plt.show()


if __name__ == "__main__":
    input_images_folder = "./Datasets/Dataset/Femurs/split/train/images"
    input_labels_folder = "./Datasets/Dataset/Femurs/split/train/labels"
    output_images_folder = "./Datasets/Dataset/Femurs/split/train/augmented_images"
    output_labels_folder = "./Datasets/Dataset/Femurs/split/train/augmented_labels"

    augment_images(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder)
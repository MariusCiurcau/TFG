import os
import random
import albumentations as A
import cv2
from albumentations import (
    Blur, RandomGamma, Rotate
)
from utils import read_label


def augment(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder, num_classes=3):
    random.seed(10)
    # Define augmentation pipeline
    transform = A.Compose([
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        Rotate(p=0.5, limit=15),
        # RandomBrightnessContrast(p=0.2),
        # ShiftScaleRotate(p=0.2, rotate_limit=5),
        #Blur(p=0.1),
        RandomGamma(p=0.1)
        # Normalize(),
        # ToTensorV2()
    ])

    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    count = {i: 0 for i in range(num_classes)}
    label_images = {i: [] for i in range(num_classes)}
    for image_file in os.listdir(input_images_folder):
        input_image_path = os.path.join(input_images_folder, image_file)
        input_label_path = os.path.join(input_labels_folder, os.path.splitext(image_file)[0] + '.txt')
        label = None

        if os.path.exists(input_label_path):
            label = read_label(input_label_path, num_classes)
        else:
            print(f"Image: {image_file}, label file not found.")

        count[label] += 1
        label_images[label].append(image_file)

    max_label = -1
    max_count = -1
    for label, n_images in count.items():
        if n_images > max_count:
            max_label = label
            max_count = n_images

    label_augmentations = {i: max_count - count[i] if count[i] != 0 else 0 for i in count.keys()}
    for label, n_augmentations in label_augmentations.items():
        for image_file in label_images[label]:
            input_image_path = os.path.join(input_images_folder, image_file)
            original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
            image = original_image[:, :, None]  # Add a channel dimension to make it 3D
            output_image_path = os.path.join(output_images_folder, f'{os.path.splitext(image_file)[0]}_0.jpg')
            cv2.imwrite(output_image_path, original_image)
            file = open(os.path.join(output_labels_folder, f'{os.path.splitext(image_file)[0]}_0.txt'), 'w+')
            file.write(str(label))
            file.close()

        copia = 1
        while n_augmentations > 0:
            for image_file in label_images[label]:
                if n_augmentations == 0:
                    break
                input_image_path = os.path.join(input_images_folder, image_file)
                original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
                image = original_image[:, :, None]  # Add a channel dimension to make it 3D

                transformed = transform(image=image)
                transformed_image = transformed["image"].squeeze(axis=-1)
                output_image_path = os.path.join(output_images_folder, f"{os.path.splitext(image_file)[0]}_{copia}.jpg")
                cv2.imwrite(output_image_path, transformed_image)
                file = open(os.path.join(output_labels_folder, f"{os.path.splitext(image_file)[0]}_{copia}.txt"), 'w+')
                file.write(str(label))
                file.close()

                n_augmentations -= 1
            copia += 1


if __name__ == "__main__":
    input_images_folder = "./Datasets/Dataset/Femurs/images/grayscale_images"
    input_labels_folder = "./Datasets/Dataset/Femurs/images/labels_fractura"
    output_images_folder = "./Datasets/Dataset/Femurs/images/augmented_images"
    output_labels_folder = "./Datasets/Dataset/Femurs/images/augmented_labels_fractura"

    num_classes = 3
    if num_classes == 2:
        n_augmentations = {0: 1, 1: 7}
    else:
        n_augmentations = {0: 0, 1: 3, 2: 3}

    augment(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder)

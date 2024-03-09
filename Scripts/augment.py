import os
import random
import albumentations as A
import cv2
from albumentations import (
    Blur, RandomGamma, Rotate
)
from utils import read_label


def augment(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder, num_classes=2):
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

    n_augmentations = {0: 1, 1: 7}
    if num_classes == 3:
        n_augmentations = {0: 2, 1: 4, 2: 4}


    for image_file in os.listdir(input_images_folder):
        input_image_path = os.path.join(input_images_folder, image_file)
        input_label_path = os.path.join(input_labels_folder, os.path.splitext(image_file)[0] + '.txt')
        label = None

        if os.path.exists(input_label_path):
            label = read_label(input_label_path)
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

        augmentations = n_augmentations[label]

        for i in range(1, augmentations + 1):
            transformed = transform(image=image)
            transformed_image = transformed["image"].squeeze(axis=-1)
            output_image_path = os.path.join(output_images_folder, "{0}_{1}.jpg".format(os.path.splitext(image_file)[0], i))
            cv2.imwrite(output_image_path, transformed_image)
            file = open(os.path.join(output_labels_folder, "{0}_{1}.txt".format(os.path.splitext(image_file)[0], i)), 'w+')
            file.write(str(label))
            file.close()


if __name__ == "__main__":
    input_images_folder = "./Datasets/Dataset/Femurs/grayscale_images"
    input_labels_folder = "./Datasets/Dataset/Femurs/labels_fractura"
    output_images_folder = "./Datasets/Dataset/Femurs/augmented_images"
    output_labels_folder = "./Datasets/Dataset/Femurs/augmented_labels_fractura"

    augment(input_images_folder, input_labels_folder, output_images_folder, output_labels_folder)

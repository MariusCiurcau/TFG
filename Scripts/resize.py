import os
import cv2


def resize(input_images_folder, output_images_folder, size=None, padding=True):
    os.makedirs(output_images_folder, exist_ok=True)

    width_max = height_max = 0
    if size is None:
        for root, dirs, files in os.walk(input_images_folder, topdown=False):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    img = cv2.imread(os.path.join(root, file))
                    h, w = img.shape[:2]
                    width_max = max(width_max, w)
                    height_max = max(height_max, h)
    else:
        height_max, width_max = size

    for root, dirs, files in os.walk(input_images_folder, topdown=False):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img = cv2.imread(os.path.join(root, file))
                h, w = img.shape[:2]
                if padding:
                    diff_vert = height_max - h
                    pad_top = diff_vert // 2
                    pad_bottom = diff_vert - pad_top
                    diff_hori = width_max - w
                    pad_left = diff_hori // 2
                    pad_right = diff_hori - pad_left
                    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                    cv2.imwrite(os.path.join(output_images_folder, file), img_padded)
                else:
                    if size is not None:
                        h, w = size
                    else:
                        h = w = max(width_max, height_max)
                    img_resized = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(output_images_folder, file), img_resized)


if __name__ == "__main__":
    input_folder = "./Datasets/Dataset/Femurs/augmented_images"
    output_folder = "./Datasets/Dataset/Femurs/resized_augmented_images"
    resize(input_folder, output_folder, False)

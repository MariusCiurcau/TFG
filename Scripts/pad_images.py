import cv2
import os

def pad_images_to_same_size(input_folder, output_folder):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lista todos los archivos en la carpeta de entrada
    files = os.listdir(input_folder)

    width_max = 0
    height_max = 0
    for file in files:
        # Verifica si el archivo es una imagen
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img = cv2.imread(os.path.join(input_folder, file)) 
            h, w = img.shape[:2]
            width_max = max(width_max, w)
            height_max = max(height_max, h)    

    for file in files:
        img = cv2.imread(os.path.join(input_folder, file)) 
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        cv2.imwrite(os.path.join(output_folder, file), img_padded)
        print(f"Saved image {file}")

if __name__ == "__main__":
    input_folder = "./Datasets/Dataset/Femurs/grayscale_images"
    output_folder = "./Datasets/Dataset/Femurs/padded_images"

    pad_images_to_same_size(input_folder, output_folder)
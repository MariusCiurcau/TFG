import os

import cv2
from PIL import Image


def estandarizar_escala_colores(input_folder, output_folder):
    # Asegúrate de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lista todos los archivos en la carpeta de entrada
    files = os.listdir(input_folder)

    for file in files:
        # Verifica si el archivo es una imagen
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Lee la imagen
            image_path = os.path.join(input_folder, file)
            original_image = cv2.imread(image_path)

            # img = Image.open(image_path)
            # img = img.convert('L')

            original_image = cv2.imread(image_path)
            grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(grayscale_image)

            # Guarda la imagen procesada en la carpeta de salida
            output_path = os.path.join(output_folder, file)
            #img.save(output_path)
            cv2.imwrite(output_path, equalized_image)


if __name__ == "__main__":
    input_folder = "../Datasets/Dataset/Femurs/flipped_images"
    output_folder = "../Datasets/Dataset/Femurs/grayscale_images"

    estandarizar_escala_colores(input_folder, output_folder)

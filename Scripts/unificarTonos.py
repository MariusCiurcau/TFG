import os

from PIL import Image


def estandarizar_escala_colores(folder_path, output_folder):
    # Asegúrate de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lista todos los archivos en la carpeta de entrada
    files = os.listdir(folder_path)

    for file in files:
        # Verifica si el archivo es una imagen
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Lee la imagen
            image_path = os.path.join(folder_path, file)
            img = Image.open(image_path)

            # Convierte la imagen a escala de grises
            img = img.convert('L')

            # Ajusta el tono para huesos y fondo
            # img = ajustar_tono(img)

            # Guarda la imagen procesada en la carpeta de salida
            output_path = os.path.join(output_folder, file)
            img.save(output_path)


def ajustar_tono(img):
    # Puedes ajustar el tono aquí según tus necesidades.
    # Por ejemplo, puedes aplicar un filtro para resaltar huesos y ajustar el fondo.
    # Aquí hay un ejemplo simple que incrementa el contraste:
    img = img.point(lambda p: p * 1.2)

    return img


if __name__ == "__main__":
    input_folder = "./Datasets/Dataset/Femurs/flipped_images"
    output_folder = "./Datasets/Dataset/Femurs/grayscale_images"

    estandarizar_escala_colores(input_folder, output_folder)

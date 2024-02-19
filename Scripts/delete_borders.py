import os
from PIL import Image

def remove_transparent_margins(image_path, output_path):
    # Abrir la imagen
    image = Image.open(image_path)

    # Obtener el cuadro delimitador de la imagen sin márgenes transparentes
    bbox = image.getbbox()

    # Recortar la imagen según el cuadro delimitador
    cropped_image = image.crop(bbox)

    # Guardar la imagen recortada
    cropped_image.save(output_path)

    print("Imagen recortada guardada en:", output_path)

def process_images_in_directory(input_directory, output_directory):
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterar sobre todos los archivos en el directorio de entrada
    for filename in os.listdir(input_directory):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Construir las rutas de entrada y salida
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Eliminar los márgenes transparentes de la imagen
            remove_transparent_margins(input_path, output_path)

if __name__ == "__main__":
    # Directorios de entrada y salida
    input_directory = "../Datasets/AO/crops"
    output_directory = "../Datasets/AO/sinBordes"

    # Procesar todas las imágenes en el directorio de entrada
    process_images_in_directory(input_directory, output_directory)

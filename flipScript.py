from PIL import Image
import os

def flip_images(folder_path, output_folder):
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

            # Voltea horizontalmente la imagen
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Guarda la imagen volteada en la carpeta de salida
            output_path = os.path.join(output_folder, file)
            flipped_img.save(output_path)

if __name__ == "__main__":
    input_folder = "/Users/quiquequeipodellano/Documents/GitHub/TFG/Datasets/Dataset/Femurs/images/flipear"
    output_folder = "Datasets/Dataset/Femurs/images/flipeadas"
    flip_images(input_folder, output_folder)

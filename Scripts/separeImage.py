import os
import shutil

# Directorio donde están las imágenes y los archivos de texto
directorio_principal = "../Datasets/Dataset/Femurs"

# Directorios de salida para cada categoría
categorias = {"0": "../Datasets/Dataset/Femurs/0",
              "1": "../Datasets/Dataset/Femurs/1",
              "2": "../Datasets/Dataset/Femurs/2"}

for archivo_label in os.listdir(os.path.join(directorio_principal, "labels_fractura_subclases")):
    if archivo_label.endswith(".txt"):
        # Obtener el nombre de la imagen correspondiente
        nombre_imagen = archivo_label[:-4]  # Eliminar la extensión .txt
        # Leer el contenido del archivo de texto
        with open(os.path.join(directorio_principal, "labels_fractura_subclases", archivo_label), 'r') as file:
            try:
                contenido = file.read().strip()
                print(f"Contenido del archivo {archivo_label}: {contenido}")
                # Verificar si el contenido es '0', '1' o '2' y copiar la imagen a la carpeta correspondiente
                if contenido in categorias:
                    # Verificar si la imagen existe antes de copiarla
                    for extension in ["jpg", "png"]:
                        ruta_imagen = os.path.join(directorio_principal, "resized_images", f"{nombre_imagen}.{extension}")
                        if os.path.exists(ruta_imagen):
                            # Copiar la imagen a la carpeta correspondiente
                            shutil.copy(ruta_imagen, categorias[contenido])
                            break
                    else:
                        print(f"No se encontró la imagen para {nombre_imagen}.")
                else:
                    print(f"El archivo {archivo_label} contiene una categoría no válida: {contenido}")
            except Exception as e:
                print(f"Error al procesar el archivo {archivo_label}: {e}")

print("Proceso completado.")

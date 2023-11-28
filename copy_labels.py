import os
import shutil

# Rutas de las carpetas
carpeta_imagenes = 'Datasets/Dataset/Proximal/Unilateral/images'
carpeta_etiquetas = 'Datasets/Proximal Femur Fracture.v11i.yolov5pytorch/train/labels'
carpeta_destino = 'Datasets/Dataset/Proximal/Unilateral/labels'

# Obtener la lista de archivos .jpg en la carpeta de imágenes
archivos_jpg = [f for f in os.listdir(carpeta_imagenes) if f.endswith('.jpg')]

# Iterar sobre los archivos .jpg
for archivo_jpg in archivos_jpg:
    # Construir la ruta completa de la imagen
    ruta_imagen = os.path.join(carpeta_imagenes, archivo_jpg)

    # Construir el nombre del archivo .txt correspondiente
    nombre_txt = os.path.splitext(archivo_jpg)[0] + '.txt'
    
    # Construir la ruta completa de la etiqueta
    ruta_etiqueta = os.path.join(carpeta_etiquetas, nombre_txt)

    # Verificar si el archivo .txt existe
    if os.path.exists(ruta_etiqueta):
        # Copiar el archivo .txt a la carpeta de destino
        shutil.copy(ruta_etiqueta, carpeta_destino)
        print(f"Se ha copiado {nombre_txt} a {carpeta_destino}")
    else:
        print(f"No se encontró etiqueta correspondiente para {archivo_jpg}")

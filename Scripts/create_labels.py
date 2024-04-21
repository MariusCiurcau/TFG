import os
import shutil

# Rutas de las carpetas
carpeta_imagenes = '/Users/quiquequeipodellano/Documents/GitHub/TFG/Datasets/FXMalaga/ULTIMAS/resized'
carpeta_destino = '/Users/quiquequeipodellano/Documents/GitHub/TFG/Datasets/FXMalaga/ULTIMAS/labels'

# Obtener la lista de archivos .jpg en la carpeta de imágenes
archivos_jpg = [f for f in os.listdir(carpeta_imagenes)]

# Iterar sobre los archivos .jpg
for archivo_jpg in archivos_jpg:
    # Construir la ruta completa de la imagen
    ruta_imagen = os.path.join(carpeta_imagenes, archivo_jpg)

    # Construir el nombre del archivo .txt correspondiente
    nombre_txt = os.path.splitext(archivo_jpg)[0] + '.txt'

    # Construir la ruta completa de la etiqueta
    ruta_etiqueta = os.path.join(carpeta_destino, nombre_txt)

    with open(ruta_etiqueta, 'w') as archivo:
        if nombre_txt.startswith('cuello'):
            archivo.write('1')
        elif nombre_txt.startswith('pertro'):
            archivo.write('2')
        elif nombre_txt.startswith('sf'):
            archivo.write('0')
        print(f"Archivo {nombre_txt} creado con éxito.")
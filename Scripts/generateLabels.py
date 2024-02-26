import os
import shutil

# Rutas de las carpetas
carpeta_imagenes = '../Datasets/FXMalaga/images'
carpeta_destino = '../Datasets/FXMalaga/labels'

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
        if nombre_txt.startswith('JC_2') or nombre_txt.startswith('JC_1'):
            archivo.write('1')
        elif nombre_txt.startswith('JC_0'):
            archivo.write('0')   
        else:
            print(f"No se ha creado label para {nombre_txt}")
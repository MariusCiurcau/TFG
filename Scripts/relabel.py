import os

# Ruta de la carpeta de entrada y salida
carpeta_entrada = '../Datasets/Dataset/Femurs/labels_fractura'
carpeta_salida = '../Datasets/Dataset/Femurs/labels_fractura_subclases'

# Asegúrate de que la carpeta de salida exista, si no, créala
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Iterar sobre los archivos en la carpeta de entrada
for nombre_archivo in os.listdir(carpeta_entrada):
    ruta_archivo_entrada = os.path.join(carpeta_entrada, nombre_archivo)
    ruta_archivo_salida = os.path.join(carpeta_salida, nombre_archivo)

    # Leer el archivo de entrada
    with open(ruta_archivo_entrada, 'r') as archivo_entrada:
        contenido = archivo_entrada.read()

    contenido_modificado = contenido
    if (contenido == '1'):
        if (nombre_archivo.startswith('inter') or nombre_archivo.startswith('subtr') or nombre_archivo.startswith('grate')):
            contenido_modificado = '2'

    # Escribir el contenido modificado en el archivo de salida
    with open(ruta_archivo_salida, 'w') as archivo_salida:
        archivo_salida.write(contenido_modificado)

    print(f"Archivo '{nombre_archivo}' procesado y guardado en la carpeta de salida.")

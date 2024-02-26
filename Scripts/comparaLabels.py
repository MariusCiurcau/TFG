import os

def obtener_nombres_sin_extension(ruta):
    """
    Obtiene una lista de nombres de archivo sin extensiones en una ruta dada.
    """
    nombres_sin_extension = []
    for archivo in os.listdir(ruta):
        nombre, extension = os.path.splitext(archivo)
        nombres_sin_extension.append(nombre)
    return nombres_sin_extension

def comparar_archivos_sin_extension(ruta1, ruta2):
    """
    Compara los nombres de archivos sin extensiones en dos rutas dadas y devuelve
    una lista de archivos que están en ruta1 pero no en ruta2.
    """
    nombres_sin_extension1 = obtener_nombres_sin_extension(ruta1)
    nombres_sin_extension2 = obtener_nombres_sin_extension(ruta2)

    archivos_faltantes = []
    for nombre in nombres_sin_extension1:
        if nombre not in nombres_sin_extension2:
            archivos_faltantes.append(nombre)
    return archivos_faltantes

# Rutas de los directorios a comparar
ruta_directorio1 = '../Datasets/Facturas de cadera IA/images'
ruta_directorio2 = '../Datasets/Facturas de cadera IA/labels'

# Comparar los nombres de archivos sin extensiones en los directorios dados
archivos_faltantes = comparar_archivos_sin_extension(ruta_directorio1, ruta_directorio2)

# Imprimir los archivos que están en ruta_directorio1 pero no en ruta_directorio2
print("Archivos en", ruta_directorio1, "pero no en", ruta_directorio2, ":")
for archivo in archivos_faltantes:
    print(archivo)

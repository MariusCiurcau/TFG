import os

def encontrar_duplicados(carpeta):
    nombres_archivos = {}
    duplicados = []

    for archivo in os.listdir(carpeta):
        nombre = os.path.basename(archivo)
        nombre_sin_extension = os.path.splitext(nombre)[0]
        if nombre_sin_extension in nombres_archivos:
            if nombres_archivos[nombre_sin_extension] is None:
                duplicados.append(nombre_sin_extension)
            else:
                duplicados.append(nombre_sin_extension)
                duplicados.append(nombres_archivos[nombre_sin_extension])
                nombres_archivos[nombre_sin_extension] = None
        else:
            nombres_archivos[nombre_sin_extension] = archivo

    return duplicados

if __name__ == "__main__":
    carpeta = "../Datasets/original_AO/crops"
    archivos_duplicados = encontrar_duplicados(carpeta)
    
    if archivos_duplicados:
        print("Archivos duplicados encontrados:")
        for archivo in archivos_duplicados:
            print(archivo)
    else:
        print("No se encontraron archivos duplicados en la carpeta.")

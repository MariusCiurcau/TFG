import os
from PIL import Image

def invertir_colores_si_no_es_negro(imagen):
    # Abrir la imagen

    img = Image.open(imagen)
    img = img.convert('RGB')

    
    # Obtener el color del pixel de la esquina superior izquierda
    color_esquina_superior_izquierda = img.getpixel((0, 0))
    # Si el color no es negro, invertir los colores de la imagen
    if color_esquina_superior_izquierda >= (160, 160, 160):
        img = Image.eval(img, lambda x: 255 - x)        
        # Guardar la imagen modificada en el directorio "Invertidas" en el mismo directorio que las imágenes originales
        ruta_guardado = os.path.join(os.path.dirname(imagen), "Invertidas", os.path.basename(imagen))
        img.save(ruta_guardado)
    
    else:
        # Si el color es negro, guardar la imagen sin cambios en el directorio "Invertidas" en el mismo directorio que las imágenes originales
        ruta_guardado = os.path.join(os.path.dirname(imagen), "Invertidas", os.path.basename(imagen))
        img.save(ruta_guardado)

def invertir(imagen):

    img = Image.open(imagen)
    img = img.convert('RGB')
    img = Image.eval(img, lambda x: 255 - x)        
    # Guardar la imagen modificada en el directorio "Invertidas" en el mismo directorio que las imágenes originales
    ruta_guardado = os.path.join(os.path.dirname(imagen), "Invertidas", os.path.basename(imagen))
    img.save(ruta_guardado)

    

def main():

    # Definir la ruta del directorio
    directorio = "./Datasets/AO/inv/"

    # Obtener la ruta completa del directorio "Invertidas"
    ruta_invertidas = os.path.join(directorio, "Invertidas")

    # Verificar si el directorio "Invertidas" existe, si no, crearlo
    if not os.path.exists(ruta_invertidas):
        os.makedirs(ruta_invertidas)

    # Obtener la lista de archivos en el directorio
    archivos = os.listdir(directorio)

    # Filtrar solo los archivos de imagen
    imagenes = [archivo for archivo in archivos if archivo.endswith((".jpg", ".jpeg", ".png"))]

    # Procesar cada imagen
    for imagen in imagenes:
        ruta_imagen = os.path.join(directorio, imagen)
        invertir(ruta_imagen)


main()

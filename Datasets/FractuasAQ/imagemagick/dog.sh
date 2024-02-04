#!/bin/bash

# Verificar si se proporciona una carpeta como argumento
if [ "$#" -ne 4 ]; then
    echo "Uso: $0 <carpeta_de_imagenes> <radius1> <radius2> <factor>"
    exit 1
fi

# Definir la carpeta de imágenes
carpeta="$1"

# Obtener los parámetros
radius1="$2"
radius2="$3"
factor="$4"

# Crear una carpeta para almacenar las imágenes procesadas
mkdir -p "${carpeta}_diferencia_gaussiana"

# Iterar sobre cada archivo de imagen en la carpeta
for imagen in "$carpeta"/*.{jpg,jpeg,png}; do
    # Obtener el nombre del archivo sin extensión
    nombre=$(basename -- "$imagen")
    nombre_sin_extension="${nombre%.*}"

    # Definir el nombre del archivo de salida
    salida="${carpeta}_diferencia_gaussiana/${nombre_sin_extension}_diferencia_gaussiana.png"

    # Ejecutar GIMP en modo de consola para aplicar la diferencia gaussiana
    /Applications/GIMP.app/Contents/MacOS/gimp -i -b "(let* ((image (car (gimp-file-load RUN-NONINTERACTIVE \"$imagen\" \"$imagen\")))
                        (drawable (car (gimp-image-get-active-layer image))))
                   (plug-in-difference-of-gaussians RUN-NONINTERACTIVE image drawable $radius1 $radius2 $factor)
                   (gimp-file-save RUN-NONINTERACTIVE image drawable \"$salida\" \"$salida\")
                   (gimp-image-delete image)
                   (gimp-quit 0))"
done

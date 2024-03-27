import os
import shutil

# Directorio de entrada y salida
input_dir = "../Datasets/original_AO/labels"
output_dir = "../Datasets/original_AO/labels2class"

# Comprueba si el directorio de salida existe, si no, créalo
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Recorre el directorio de entrada
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r') as file:
            content = file.read()
            
        # Reemplaza los 2 con 1 y escribe en el nuevo archivo
        new_content = content.replace('2', '1')
        
        # Escribe el nuevo contenido en el archivo de salida
        with open(output_path, 'w') as file:
            file.write(new_content)
        
        print(f"Archivo procesado: {input_path}")

print("Proceso completado.")


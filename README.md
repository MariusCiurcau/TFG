# XAI for Hip Fracture Recognition

Este repositorio contiene el código fuente del Trabajo de Fin de Grado (TFG) titulado **"Inteligencia Artificial Explicable para el reconocimiento de fracturas de cadera"** (*XAI for Hip Fracture Recognition*).

## Descripción

El objetivo de este proyecto es desarrollar un sistema de inteligencia artificial que sea capaz de reconocer fracturas de cadera de manera precisa y explicable. La explicabilidad es un aspecto clave del proyecto, ya que permite entender y confiar en las decisiones tomadas por el modelo de IA.

## Autores

Este trabajo ha sido desarrollado por:

- **Enrique Queipo de Llano Burgos**
- **Alejandro Paz Olalla**
- **Marius Ciurcau**

## Directores

El proyecto ha sido dirigido por:
- **Belén Díaz Agudo**
- **Juan A. Recio García**

## Información Adicional

- **Convocatoria:** Junio 2024
- **Doble Grado en Ingeniería Informática y Matemáticas**
- **Facultad de Informática, Universidad Complutense de Madrid**

## Estructura del Repositorio

- `Datasets/`: conjuntos de datos utilizados.
- `Reports/`: informes de entrenamiento y evaluación de modelos.
- `Scripts/`: scripts para la ejecución y preprocesamiento de datos.
- `experiment1/`, `experiment2/`, `experiment3/`: directorios de los experimentos realizados.
- `figures/`: figuras y gráficos generados durante el proyecto.
- `models/`: modelos entrenados.
- `yolov5_ws/`: modelo ajustado de YOLOv5.

## Instrucciones de Uso

1. Clona el repositorio:
   
   ```bash
   git clone https://github.com/MariusCiurcau/TFG-XAI-for-Hip-Fracture-Recognition.git
3. Navega al directorio del proyecto:
   
   ```bash
   cd TFG-XAI-for-Hip-Fracture-Recognition
4. Instala las dependencias:
   
   ```bash
   pip install -r requirements.txt

## Uso de Funcionalidades de GPT-4
Para habilitar las funcionalidades de GPT-4, sigue estos pasos:

1. Crea un archivo `key.txt` en el directorio raíz del repositorio y coloca tu clave de API de OpenAI dentro del archivo.
2. En `model.py`, establece la variable `USE_GPT = True`.

En caso de que no se disponga de una clave de API de OpenAI o no se quiera hacer uso de GPT-4 se debe mantener `USE_GPT = False`

## Realizar inferencia con `model.py`
Para utilizar el script `model.py` con la opción `--predict`, sigue estos pasos:

1. Navega al directorio `Scripts/`.
2. Ejecuta el siguiente comando, reemplazando las rutas y valores según sea necesario:
   
   ```bash
   python model.py --predict --load <ruta_del_modelo> --image <ruta_de_la_imagen> --labels <ruta_de_las_etiquetas> --num_classes <numero_de_clases>

- `--load <ruta_del_modelo>`: ruta al archivo del modelo entrenado. Los nombre de los modelos en `models/` contienen información sobre el número de clases consideradas por el modelo, el número de iteraciones del proceso de entrenamiento y el conjunto de datos utilizado. Los nombres son de la forma `resnet18_<numero_de_iteraciones>_<numero_de_clases>_<subconjuntos_de_datos_utilizados>`, donde los subconjuntos de datos utilizados son una combinación de los datasets `ROB`, `AO` y `HVV`.
- `--image <ruta_de_la_imagen>`: ruta a la imagen que deseas clasificar o a un directorio de imágenes. Deben ser imágenes de dimensiones 224 x 224 y en escala de grises.
- `--labels <ruta_de_las_etiquetas>`: ruta al archivo de etiquetas correspondiente a las clases del modelo. Este parámetro es opcional y solo se utiliza cuando se especifica una ruta a un directorio de imágenes. Las etiquetas deben estar contenidas en archivos de texto con el mismo nombre que las imágenes.
- `--num_classes <numero_de_clases>`: número de clases para la predicción. Debe coincidir con el número de clases con el que ha sido entrenado el modelo.

Ejemplo de uso:

```bash
python model.py --predict --load ../models/resnet18_10_2_ROB_AO_HVV --image Datasets/ROB/images/ROB_0327.jpg --num_classes 2

import os

import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, Occlusion, Rise, Lime, KernelShap, SobolAttributionMethod)
from xplique.metrics import Deletion, Insertion
from xplique.plots import plot_attributions
from xplique.wrappers import TorchWrapper

tf.config.run_functions_eagerly(True)


img_list = [
    ('/Users/quiquequeipodellano/Downloads/ImagenesMemoria/ROB/images/ROB_0047.jpg',1),
    ('/Users/quiquequeipodellano/Downloads/ImagenesMemoria/ROB/images/ROB_0056.jpg',1),
    ('/Users/quiquequeipodellano/Downloads/ImagenesMemoria/AO/images/AO_0038.jpg',1),
    ('/Users/quiquequeipodellano/Downloads/ImagenesMemoria/AO/images/AO_0012.jpg',1),
    ('/Users/quiquequeipodellano/Downloads/ImagenesMemoria/AO/images/AO_0017.jpg',1)
]

X = []
Y = []

for img_name, label in img_list:
    print(img_name)
    img = cv2.imread(img_name)[..., ::-1] # when cv2 load an image, the channels are inversed
    label = tf.keras.utils.to_categorical(label, 2)
    X.append(img)
    Y.append(label)

#X = np.array(X, dtype=np.float32) # in the Getting Started tutorial
X = np.array(X, dtype=np.uint8) # slight change here
Y = np.array(Y)

plt.rcParams["figure.figsize"] = [15, 6]
for img_id, img in enumerate(X):
  plt.subplot(1, len(X), img_id+1)
  plt.imshow(img)
  # plt.imshow(img/255.0) # as img is now a uint8 that is not necessary
  plt.axis('off')
plt.savefig("../figures/xplique/images")
plt.show()

X_float = np.empty((len(X), 3, 224, 224), dtype=np.float32)

def predict_xplique(load_path, width, height):
    #image = Image.open(image_path)
    #transform = transforms.ToTensor()
    #input_image = transform(image)
    #print(input_image.shape)
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # para resnet
    #model = ConvNet(input_size, output_size, channels)


    model.load_state_dict(torch.load(load_path))

    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapped_model = TorchWrapper(model, device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for i, x in enumerate(X):
        X_float[i] = preprocess(x)
    #print('X_float[0]', X_float[0])
    X_preprocessed = torch.tensor(X_float, dtype=torch.float32)
    #X_preprocessed = torch.from_numpy(X_float)
    #X_preprocessed = torch.stack([preprocess(x) for x in X])
    #print(X_preprocessed[0])
    X_preprocessed4explainer = np.moveaxis(X_preprocessed.numpy(), [1, 2, 3], [3, 1, 2])
    # set batch size parameter
    batch_size = 64

    # build the explainers
    explainers = [
                Saliency(wrapped_model),
                GradientInput(wrapped_model),
                IntegratedGradients(wrapped_model, steps=80, batch_size=batch_size),
                SmoothGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
                SquareGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
                VarGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
                Occlusion(wrapped_model, patch_size=10, patch_stride=5, batch_size=batch_size),
                Rise(wrapped_model, nb_samples=4000, batch_size=batch_size),
                SobolAttributionMethod(wrapped_model, batch_size=batch_size),
                Lime(wrapped_model, nb_samples = 4000, batch_size=batch_size),
                KernelShap(wrapped_model, nb_samples = 4000, batch_size=batch_size)
    ]
    
    n_explainers = len(explainers)
    n_test_img = len(img_list)

    deletion_scores = {}
    insertion_scores = {}

    for i, explainer in enumerate(explainers):

        explanations = explainer(X_preprocessed4explainer, Y)
        name = explainer.__class__.__name__
        deletion = Deletion(wrapped_model, X_preprocessed4explainer, Y)
        insertion = Insertion(wrapped_model, X_preprocessed4explainer, Y)
        deletion_score = deletion(explanations)
        insertion_score = insertion(explanations)
        deletion_scores[name] = deletion_score
        insertion_scores[name] = insertion_score
        print(f"Method: {name}")
        print(f"Deletion: {deletion_score}")
        print(f"Insertion: {insertion_score}")
        plot_attributions(explanations, X, img_size=2., cmap='jet', alpha=0.4,
                            cols=len(X), absolute_value=True, clip_percentile=0.5)
        
        print("\n")
        plt.savefig(f"../figures/xplique/explanations/{name}")
    
    # Obtener nombres y puntuaciones para graficar
    explainer_names = list(deletion_scores.keys())
    deletion_values = list(deletion_scores.values())
    insertion_values = list(insertion_scores.values())

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según tu preferencia

    plt.bar(explainer_names, deletion_values, color='skyblue')

    # Añadir etiquetas y título
    plt.xlabel('Explainers')
    plt.ylabel('Deletion Scores')
    plt.title('Deletion Scores for Each Explainer')

    # Rotar etiquetas del eje x para mejor legibilidad si son largas
    plt.xticks(rotation=45, ha='right')

    # Mostrar el gráfico
    plt.tight_layout()  # Ajustar el diseño para que las etiquetas no se solapen
    plt.savefig(f"../figures/xplique/deletion-scores")

    plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según tu preferencia

    plt.bar(explainer_names, insertion_values, color='skyblue')

    # Añadir etiquetas y título
    plt.xlabel('Explainers')
    plt.ylabel('Insertion Scores')
    plt.title('Insertion Scores for Each Explainer')

    # Rotar etiquetas del eje x para mejor legibilidad si son largas
    plt.xticks(rotation=45, ha='right')

    # Mostrar el gráfico
    plt.tight_layout()  # Ajustar el diseño para que las etiquetas no se solapen
    plt.savefig(f"../figures/xplique/insertion-scores")


def create_table():
    image_folder = "../figures/xplique/explanations"
    image_names = os.listdir(image_folder)

    num_rows = len(image_names) + 1

    # Create subplots
    fig, ax = plt.subplots(num_rows, 2, figsize=(8, num_rows*4))  # Adjust size as needed

    image = Image.open('../figures/xplique/images.png')
    ax[0,1].imshow(image)
    ax[0,1].axis('off')  # Turn off axis
        
    # Set title as the image name
    ax[0,0].text(0.5,0.5,'Images')
    ax[0,0].axis('off')

    # Iterate over image names and display images
    for i, image_name in enumerate(image_names):
        # Construct the full path to the image
        image_path = os.path.join(image_folder, image_name)
        
        # Open and display the image
        image = Image.open(image_path)
        ax[i+1,1].imshow(image)
        ax[i+1,1].axis('off')  # Turn off axis
        
        # Set title as the image name
        ax[i+1,0].text(0.5,0.5,os.path.splitext(image_name)[0])

        ax[i+1,0].axis('off')  # Add a placeholder to the right for alignment

    # Adjust layout and display the plot
    plt.subplots_adjust(hspace=0.01)  # Adjust vertical spacing between subplots
    plt.tight_layout()
    plt.savefig(f"../figures/xplique/table")
    plt.show()

predict_xplique(load_path='../models/resnet18_10_2_ROB_AO_HVV', width=224, height=224)
#create_table()
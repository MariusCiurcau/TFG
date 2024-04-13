import os
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms

import xplique
from xplique.metrics import Deletion
from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, Occlusion, Rise, GuidedBackprop, Lime, KernelShap, SobolAttributionMethod)

tf.config.run_functions_eagerly(True)


img_list = [
    ('../Datasets/Dataset/Femurs/clusters/label0/ROB_0017_0.jpg',0),
    ('../Datasets/Dataset/Femurs/clusters/label0/ROB_0066_0.jpg',0),
    ('../Datasets/Dataset/Femurs/clusters/label1/ROB_0053_0.jpg',1),
    ('../Datasets/Dataset/Femurs/clusters/label1/ROB_0056_0.jpg',1),
    ('../Datasets/Dataset/Femurs/clusters/label2/ROB_0002_0.jpg',1)
]

X = []
Y = []

for img_name, label in img_list:

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

    scores = {}

    for i, explainer in enumerate(explainers):

        explanations = explainer(X_preprocessed4explainer, Y)
        name = explainer.__class__.__name__
        metric = Deletion(wrapped_model, X_preprocessed4explainer, Y)
        score = metric(explanations)
        scores[name] = score
        print(f"Method: {name}")
        print(f"Score: {score}")
        plot_attributions(explanations, X, img_size=2., cmap='jet', alpha=0.4,
                            cols=len(X), absolute_value=True, clip_percentile=0.5)
        
        print("\n")
        plt.savefig(f"../figures/xplique/{name}")
    
    print(scores)

    plt.show()

predict_xplique(load_path='../models/resnet18_10_2_ROB', width=224, height=224)
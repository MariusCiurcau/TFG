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
from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, Occlusion, Rise, GuidedBackprop, Lime, KernelShap, SobolAttributionMethod)

tf.config.run_functions_eagerly(True)

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),

            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(16 * input_size, output_size),
            # nn.Softmax(dim=1)
        )

    def forward(self, x_batch):
        preds = self.seq(x_batch)
        return preds

img_list = [
    ('../Datasets/Dataset/Femurs/resized_images/normal_305_png.rf.1d784b164386cda78ef3556a87f5890b_0.jpg',0),
    ('../Datasets/Dataset/Femurs/resized_images/normal_264_png.rf.179d1b299132ad5888c3a93a20af58b02_1.jpg',0),
    ('../Datasets/Dataset/Femurs/resized_images/neck_73_png.rf.dc2268059fe3bec20b168fb13f35b3162_8.jpg',1),
    ('../Datasets/Dataset/Femurs/resized_images/intertrochanteric_59_png.rf.f0cc5f2cf342401d2a4be25b93fe231f_7.jpg',1)
]
image_path = '../Datasets/Dataset/Femurs/resized_images/neck_73_png.rf.dc2268059fe3bec20b168fb13f35b3162_8.jpg'

X = []
Y = []

for img_name, label in img_list:
    img = cv2.imread(img_name)[..., ::-1] # when cv2 load an image, the channels are inversed
    label = tf.keras.utils.to_categorical(label, 2)

    X.append(img)
    Y.append(label)

# X = np.array(X, dtype=np.float32) # in the Getting Started tutorial
X = np.array(X, dtype=np.uint8) # slight change here
Y = np.array(Y)

plt.rcParams["figure.figsize"] = [15, 6]
for img_id, img in enumerate(X):
  plt.subplot(1, len(X), img_id+1)
  plt.imshow(img)
  # plt.imshow(img/255.0) # as img is now a uint8 that is not necessary
  plt.axis('off')
plt.show()

def predict_xplique(load_path, width, height):
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    input_image = transform(image)
    print(input_image.shape)
    
    model = ConvNet(width * height, 2)
    model.load_state_dict(torch.load(load_path))

    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapped_model = TorchWrapper(model, device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    X_preprocessed = torch.stack([preprocess(x) for x in X])

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
                #  Lime(wrapped_model, nb_samples = 4000, batch_size=batch_size),
                #  KernelShap(wrapped_model, nb_samples = 4000, batch_size=batch_size)
    ]
    
    for explainer in explainers:

        explanations = explainer(X_preprocessed4explainer, Y)

        print(f"Method: {explainer.__class__.__name__}")
        plot_attributions(explanations, X, img_size=2., cmap='jet', alpha=0.4,
                            cols=len(X), absolute_value=True, clip_percentile=0.5)
        plt.show()
        print("\n")
    

predict_xplique(load_path='../models/rgb.pt', width=299, height=299)
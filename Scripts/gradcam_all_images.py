import os
import random

import cv2
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

#random.seed(42)
torch.manual_seed(0)

preprocess = transforms.Compose([
    #transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_rgb = transforms.Compose([
    #transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def custom_sort_key(model_name):
    parts = model_name.rsplit('_', 1)
    numeric_part = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else float('inf')
    return numeric_part

def visualize_label(visualization, label, prediction, model=None):
    if label is not None:
        visualization = cv2.putText(visualization, f"Class: {label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    if prediction is not None:
        visualization = cv2.putText(visualization, f"Prediction: {prediction}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if model is not None:
        visualization = cv2.putText(visualization, model, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return visualization


def add_border(visualization, label, pred):
    false_positive = (label == 0) and (pred != 0)
    false_negative = (label != 0) and (pred == 0)
    correct = label == pred
    if false_positive:
        visualization = cv2.copyMakeBorder(visualization, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 153, 0))
    elif false_negative:
        visualization = cv2.copyMakeBorder(visualization, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    elif correct:
        visualization = cv2.copyMakeBorder(visualization, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    return visualization


def show_gradcam(model_path, weights):
    image_dir = '../Datasets/Dataset/Femurs/resized_images'
    label_dir = '../Datasets/Dataset/Femurs/augmented_labels_fractura'

    BATCH_SIZE = 80
    N_COLS = 10
    N_ROWS = BATCH_SIZE // N_COLS

    image_files = [image for image in os.listdir(image_dir) if image.endswith('_0.jpg')]

    model = torch.hub.load('pytorch/vision:v0.10.0', model='resnet18', weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    target_layers = [model.layer4[-1]]  # especifico de resnet
    gradcam = GradCAM(model, target_layers)  # Choose the last convolutional layer
    model.eval()

    for batch in range(len(image_files) // BATCH_SIZE + 1):
        images = []
        visualizations = []
        fig, axes = plt.subplots(2 * N_ROWS, N_COLS, figsize=(20, 3 * (2 * N_ROWS)), sharex=True, dpi=300)
        plt.subplots_adjust(wspace=0, hspace=0)

        for image_path in image_files[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]:
            image_name, _ = os.path.splitext(image_path)
            label_file = os.path.join(label_dir, image_name + '.txt')
            image = Image.open(image_dir + '/' + image_path)
            images.append(image)
            rgb_image = Image.open(image_dir + '/' + image_path)

            input_image = preprocess(image).unsqueeze(0)
            rgb_input_image = preprocess_rgb(rgb_image).permute(1, 2, 0).numpy()
            output = model(input_image)
            pred = torch.argmax(output, 1)[0].item()

            with open(label_file, 'r') as file:
                label = int(file.read())

            attributions = gradcam(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
            attribution = attributions[0, :]
            if label != 0:
                visualization = show_cam_on_image(rgb_input_image, attribution, use_rgb=True)
            else:
                visualization = np.array(image)
            #if (label == 0 and pred == 1): print(image_path)
            visualization = visualize_label(visualization, label, pred)
            visualization = add_border(visualization, label, pred)
            visualizations.append(visualization)

        for i in range(min(N_ROWS, len(image_files) // N_COLS + 1)):
            for j in range(min(N_COLS, len(image_files) - i * N_COLS)):
                axes[2*i, j].imshow(images[min(N_COLS, len(image_files) - i * N_COLS)*i + j])
                axes[2*i + 1, j].imshow(visualizations[min(N_COLS, len(image_files) - i * N_COLS)*i + j]) # TODO corregir error ultima iteracion

                axes[2*i, j].axis('off')
                axes[2*i, j].get_xaxis().set_visible(False)
                axes[2*i, j].get_yaxis().set_visible(False)

                axes[2*i + 1, j].axis('off')
                axes[2*i + 1, j].get_xaxis().set_visible(False)
                axes[2*i + 1, j].get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(f'../figures/gradcam18_edge_batch_{batch}.png', dpi=600)


if __name__ == "__main__":
    show_gradcam('../models/resnet18_edge_50', weights='ResNet18_Weights.DEFAULT')
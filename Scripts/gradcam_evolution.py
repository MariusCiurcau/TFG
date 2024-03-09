import os
import random

import cv2
import torch.nn as nn
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import read_label

random.seed(42)
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
        visualization = cv2.putText(visualization, f"Clase: {label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    if prediction is not None:
        visualization = cv2.putText(visualization, f"Prediccion: {prediction}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if model is not None:
        visualization = cv2.putText(visualization, model, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return visualization


def show_gradcam(model_name, weights):
    num_classes = 2
    image_dir = '../Datasets/Dataset/Femurs/resized_images'
    label_dir = '../Datasets/Dataset/Femurs/augmented_labels_fractura'
    model_dir = '../models'
    models = [os.path.join(model_dir, file) for file in os.listdir(model_dir) if file.startswith(model_name)]
    models.sort(key=custom_sort_key)

    image_files = os.listdir(image_dir)
    random.shuffle(image_files)

    fig, axes = plt.subplots(1 + len(models), 5, figsize=(20, 3 + 3*len(models)), sharex=True, dpi=300)
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(5):
        image_path = image_files[i]
        image_name, _ = os.path.splitext(image_path)
        label_file = os.path.join(label_dir, image_name + '.txt')
        image = Image.open(image_dir + '/' + image_path)
        #with open(label_file, 'r') as file:
        #    label = file.read()
        axes[0, i].imshow(image, cmap='gray')  # Assuming images are grayscale
        axes[0, i].axis('off')


    for i in range(len(models)):
        model = torch.hub.load('pytorch/vision:v0.10.0', model='resnet34', weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load(models[i]))
        target_layers = [model.layer4[-1]]  # especifico de resnet
        gradcam = GradCAM(model, target_layers)  # Choose the last convolutional layer
        model.eval()


        visualizations = []

        for image_path in image_files[:5]:
            image_name, _ = os.path.splitext(image_path)
            label_file = os.path.join(label_dir, image_name + '.txt')
            image = Image.open(image_dir + '/' + image_path)
            rgb_image = Image.open(image_dir + '/' + image_path)

            input_image = preprocess(image).unsqueeze(0)
            rgb_input_image = preprocess_rgb(rgb_image).permute(1, 2, 0).numpy()
            output = model(input_image)
            pred = torch.argmax(output, 1)[0].item()
            label = read_label(label_file, num_classes)

            attributions = gradcam(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
            attribution = attributions[0, :]
            visualization = show_cam_on_image(rgb_input_image, attribution, use_rgb=True)
            visualization = visualize_label(visualization, label, pred)
            visualizations.append(visualization)

        for j in range(len(visualizations)):
            axes[i + 1, j].imshow(visualizations[j])
            axes[i + 1, j].get_xaxis().set_visible(False)
            axes[i + 1, j].get_yaxis().set_visible(False)

        axes[i + 1, 0].get_yaxis().set_visible(True)
        axes[i + 1, 0].set_yticks([0.5 * visualizations[0].shape[0]])
        axes[i + 1, 0].set_yticklabels([os.path.basename(models[i])], fontsize=18)

    plt.tight_layout()
    plt.savefig('../figures/gradcam_evolution.png', dpi=600)
    #plt.show()


if __name__ == "__main__":
    model_name = 'resnet34_50'

    if model_name == 'resnet18':
        weights = 'ResNet18_Weights.DEFAULT'
    #else:
        #raise ValueError('Model not supported')
    weights = 'ResNet34_Weights.DEFAULT'

    show_gradcam(model_name=model_name, weights=weights)
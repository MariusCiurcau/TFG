import os
import random

import cv2
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils import find_similar_images, add_border, visualize_label, read_label

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


def show_gradcam(model_path, weights):
    num_classes = 2
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

    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    i = 0
    for batch in range(len(image_files) // BATCH_SIZE + 1):
        images = []
        visualizations = []
        fig, axes = plt.subplots(2 * N_ROWS, N_COLS, figsize=(20, 3 * (2 * N_ROWS)), sharex=True, dpi=300)
        plt.subplots_adjust(wspace=0, hspace=0)

        for image_path in image_files[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]:
            i += 1
            image_name, _ = os.path.splitext(image_path)
            print(i, ': ', image_name)
            label_file = os.path.join(label_dir, image_name + '.txt')
            image = Image.open(image_dir + '/' + image_path)
            images.append(image)
            rgb_image = Image.open(image_dir + '/' + image_path)

            input_image = preprocess(image).unsqueeze(0)
            rgb_input_image = preprocess_rgb(rgb_image).permute(1, 2, 0).numpy()
            output = model(input_image)
            pred = torch.argmax(output, 1)[0].item()
            label = read_label(label_file)

            metric_targets = [ClassifierOutputSoftmaxTarget(pred)]

            if label != 0:
                attributions = gradcam(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
                attribution = attributions[0, :]
                scores = cam_metric(input_image, attributions, metric_targets, model)
                score = scores[0]
                visualization = show_cam_on_image(rgb_input_image, attribution, use_rgb=True)
                visualization = visualize_label(visualization, label, pred, score=score)
            else:
                similar_image = find_similar_images(image_path, label, image_files, image_dir, label_dir, num_images=1)[0]
                visualization = np.array(cv2.imread(image_dir + '/' + similar_image))
                visualization = visualize_label(visualization, label, pred, similar=True)
            visualization = add_border(visualization, label, pred)
            #if (label == 0 and pred == 1): print(image_path)
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
        plt.savefig(f'../figures/gradcam18_10_2_AO_AQ_batch_{batch}.png', dpi=600)


if __name__ == "__main__":
    show_gradcam('../models/resnet18_10_AO_AQ', weights='ResNet18_Weights.DEFAULT')
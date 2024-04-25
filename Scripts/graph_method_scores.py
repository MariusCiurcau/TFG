import argparse
import os
import random
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam.metrics.road import ROADCombined

import torch.nn as nn
import torch
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, \
    EigenGradCAM, RandomCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from xplique.attributions import Rise
from xplique.wrappers import TorchWrapper
from torchvision import transforms

from utils import read_label

torch.manual_seed(0)

def compute_method_scores(load_path, num_classes):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # para resnet
    model.load_state_dict(torch.load(load_path))
    target_layers = [model.layer4[-1]]  # especifico de resnet
    model.eval()

    image_dir = '../Datasets/COMBINED/resized_images'
    label_dir = '../Datasets/COMBINED/augmented_labels'


    image_files_aux = os.listdir(image_dir)
    image_files_aux = [image_file for image_file in image_files_aux if image_file.endswith('_0.jpg')]
    image_files = {}

    for image_path in image_files_aux:
        image_name, _ = os.path.splitext(image_path)
        label_file = os.path.join(label_dir, image_name + '.txt')
        label = read_label(label_file, num_classes)
        if label != 0:
            image_files[image_path] = label


    methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers)),
               ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers)),
               ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers)),
               ("AblationCAM", AblationCAM(model=model, target_layers=target_layers)),
               ("RandomCAM", RandomCAM(model=model, target_layers=target_layers))]
    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    classes = list(range(1, num_classes))
    names = [name for name, _ in methods]
    scores = {name: {clase: .0 for clase in classes} for name in names}

    image_count = 0
    i = 0

    #image_files = image_files[:5] # para probar

    for image_path, label in image_files.items():
        i += 1
        print('Imagen', i, '/', len(image_files), ':', image_path)
        image_name, _ = os.path.splitext(image_path)
        #label_file = os.path.join(label_dir, image_name + '.txt')
        image = Image.open(image_dir + '/' + image_path)

        input_image = preprocess(image).unsqueeze(0)
        #rgb_input_image = preprocess(image).permute(1, 2, 0).numpy()
        output = model(input_image)
        pred = torch.argmax(output, 1)[0].item()

        #with open(label_file, 'r') as file:
            #label = int(file.read())
        print('Etiqueta:', label, 'Predicción:', pred)

        metric_targets = [ClassifierOutputSoftmaxTarget(pred)]

        if label == pred:
            image_count += 1
            for name, cam_method in methods:
                attributions = cam_method(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
                #attribution = attributions[0, :]
                score = cam_metric(input_image, attributions, metric_targets, model)
                score = score[0]
                scores[name][label] += score

    with open('method_metrics.pkl', 'wb') as f:
        pickle.dump(scores, f)

    for name, dict in scores.items():
        scores[name] = {k: v / image_count for k, v in dict.items()}

    return scores, names, classes

def plot_method_scores(scores, names, classes):
    fig, ax = plt.subplots()
    ax.bar(names, [scores[name][clase] for name in names for clase in classes])
    ax.set_ylabel('Avg. ROADCombined Score')
    ax.set_xlabel('Method')
    plt.savefig('../figures/method_scores.png')

if __name__ == "__main__":
    scores, names, classes = compute_method_scores('../models/resnet18_10_2_ROB_AO_HVV', num_classes=2)
    print('Scores 2 classes:', scores)
    print('Names 2 classes:', names)
    print('Classes 2 classes:', classes)
    #plot_method_scores(scores, names, classes)

    scores, names, classes = compute_method_scores('../models/resnet18_10_3_ROB_AO_HVV', num_classes=3)
    print('Scores 3 classes:', scores)
    print('Names 3 classes:', names)
    print('Classes 3 classes:', classes)


import argparse
import os
import random
import pickle

import matplotlib
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


rc_params = {
    "text.usetex": True,
    "font.size": 18,
    "font.family": "sans-serif",
    "text.latex.preamble": r'\usepackage[T1]{fontenc}'
}
matplotlib.rcParams.update(rc_params)
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

def plot_method_scores(scores2, scores3, classes2, classes3, names, savefig):
    n_methods = len(names)
    colors = {'specific1': 'darkgreen', 'combined1': '#66a266', 'specific2': 'darkblue', 'combined2': '#6666b9'}

    bar_width = 2
    num_bars = 3
    model_space = 0.8
    method_space = 1.2 * bar_width
    class_space = 0.2
    index = np.arange(n_methods) * (bar_width * num_bars + model_space + class_space + method_space)

    max_height = 0

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (method, classes) in enumerate(scores2.items()):
        for j, (cls, score) in enumerate(classes.items()):
            label = f'Modelo de 2 clases' if i ==0 and j == 0 else None
            pos_x = index[i] + j * (2 + model_space) * bar_width
            bars = ax.bar([pos_x], [score], bar_width, label=label, color='darkgreen')
            for bar in bars:
                height = max(bar.get_height(), 0)
                max_height = max(max_height, height)
                ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=16)

    for i, (method, classes) in enumerate(scores3.items()):
        for j, (cls, score) in enumerate(classes.items()):
            label = f'Modelo de 3 clases' if i == 0 and j == 0 else None
            pos_x = index[i] + bar_width + model_space + j * (bar_width + class_space)
            bars = ax.bar([pos_x], [score], bar_width, label=label, color='darkblue')
            for bar in bars:
                height = max(bar.get_height(), 0)
                max_height = max(max_height, height)
                ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=16)

    ax.tick_params(axis='x', which='both', direction='in', length=0, pad=35)
    ax.tick_params(axis='y', which='minor', direction='in', length=0, pad=35)
    ax.set_ylim(0, max_height * 1.1)
    ax.set_title('ROADCombined promedio por clase y método de explicación visual', pad=20)
    ax.set_xticks([x + (num_bars * bar_width + model_space + class_space) / 2 - bar_width / 2 for x in index])
    #ax.set_xticks([- bar_width / 2 + x + 2 * bar_width + model_space / 2 for x in index])
    ax.set_xticklabels(names)
    ax.legend()

    class1_ticks_pos = index
    for tickpos in class1_ticks_pos:
        plt.text(tickpos, -0.007, '1', ha='center', fontsize=14)

    class1_ticks_pos = [x + bar_width + model_space for x in index]
    for tickpos in class1_ticks_pos:
        plt.text(tickpos, -0.007, '1', ha='center', fontsize=14)

    class2_ticks_pos = [x + bar_width + class_space for x in class1_ticks_pos]
    for tickpos in class2_ticks_pos:
        plt.text(tickpos, -0.007, '2', ha='center', fontsize=14)

    ax.set_ylabel('ROADCombined promedio', labelpad=10)
    ax.set_xlabel('Método', labelpad=15)
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.25)





    if savefig is not None:
        plt.savefig(savefig, dpi=600)
    plt.show()

if __name__ == "__main__":
    """
    scores, names, classes = compute_method_scores('../models/resnet18_10_2_ROB_AO_HVV', num_classes=2)
    print('Scores 2 classes:', scores)
    print('Names 2 classes:', names)
    print('Classes 2 classes:', classes)
    #plot_method_scores(scores, names, classes)

    scores, names, classes = compute_method_scores('../models/resnet18_10_3_ROB_AO_HVV', num_classes=3)
    print('Scores 3 classes:', scores)
    print('Names 3 classes:', names)
    print('Classes 3 classes:', classes)
    """
    scores2 = {'GradCAM': {1: 0.05988918544679153}, 'GradCAM++': {1: 0.021020159337336454}, 'EigenGradCAM': {1: -0.0059206970111747095}, 'AblationCAM': {1: 0.047470008174517776}, 'RandomCAM': {1: -0.0018458233964515896}}
    names = ['GradCAM', 'GradCAM++', 'EigenGradCAM', 'AblationCAM', 'RandomCAM']
    classes2 = [1]

    scores3 = {'GradCAM': {1: 0.07248435414854497, 2: 0.14768073174156368}, 'GradCAM++': {1: 0.05772292231106096, 2: 0.14047888941353276}, 'EigenGradCAM': {1: 0.021939895396667814, 2: 0.15140482876449823}, 'AblationCAM': {1: 0.06419089309398145, 2: 0.14281609040936308}, 'RandomCAM': {1: -0.02222342440296733, 2: 0.07011932143498034}}
    classes3 = [1, 2]
    plot_method_scores(scores2, scores3, classes2, classes3, names, savefig='../figures/method_scores.pdf')


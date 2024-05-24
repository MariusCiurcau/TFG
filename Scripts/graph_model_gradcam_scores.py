import os
import pickle
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
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


def compute_model_gradcam_scores():
    num_classes = 3

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

    random.seed(0)
    keys = random.sample(list(image_files.keys()), 50)
    image_files = {key: image_files[key] for key in keys} # subsample

    combined_model_path = "../models/resnet18_10_3_ROB_AO_HVV"
    combined_model_name = os.path.basename(os.path.normpath(combined_model_path))
    combined_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    num_features = combined_model.fc.in_features
    combined_model.fc = nn.Linear(num_features, num_classes)  # para resnet
    combined_model.load_state_dict(torch.load(combined_model_path))
    combined_target_layers = [combined_model.layer4[-1]]  # especifico de resnet
    combined_model.eval()

    combined_cam_method = GradCAM(model=combined_model, target_layers=combined_target_layers)

    models = ["../models/resnet18_10_3_ROB", "../models/resnet18_10_3_ROB_AO", "../models/resnet18_10_3_HVV"]
    datasets = ["../Datasets/ROB", "../Datasets/AO", "../Datasets/HVV"]
    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    classes = list(range(1, num_classes))
    model_names = [os.path.basename(os.path.normpath(name)) for name in models]
    dataset_names = [os.path.basename(os.path.normpath(dataset)) for dataset in datasets]
    scores = {dataset_name: {model_name: {clase: .0 for clase in classes}, combined_model_name: {clase: .0 for clase in classes}} for dataset_name, model_name in zip(dataset_names, model_names)}



    for dataset_path, dataset_name, specific_model_path, specific_model_name in zip(datasets, dataset_names, models, model_names):
        print('Dataset:', dataset_name, '- Model:', specific_model_name)

        dataset_image_files = [img_path for img_path in image_files if img_path.startswith(dataset_name)]
        dataset_image_files = {img_path: image_files[img_path] for img_path in dataset_image_files}
        print(dataset_image_files)

        specific_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
        num_features = specific_model.fc.in_features
        specific_model.fc = nn.Linear(num_features, num_classes)  # para resnet
        specific_model.load_state_dict(torch.load(specific_model_path))
        specific_target_layers = [specific_model.layer4[-1]]  # especifico de resnet
        specific_model.eval()

        specific_cam_method = GradCAM(model=specific_model, target_layers=specific_target_layers)

        i = 0
        specific_image_count = 0
        combined_image_count = 0

        for image_path, label in dataset_image_files.items():
            i += 1
            print('Imagen', i, '/', len(dataset_image_files), ':', image_path)
            image_name, _ = os.path.splitext(image_path)
            image = Image.open(image_dir + '/' + image_path)

            input_image = preprocess(image).unsqueeze(0)

            specific_output = specific_model(input_image)
            specific_pred = torch.argmax(specific_output, 1)[0].item()
            combined_output = combined_model(input_image)
            combined_pred = torch.argmax(combined_output, 1)[0].item()
            print('\tEtiqueta:', label)
            print('\tPredicción modelo específico:', specific_pred)
            print('\tPredicción modelo combinado:', combined_pred)

            specific_metric_targets = [ClassifierOutputSoftmaxTarget(specific_pred)]
            combined_metric_targets = [ClassifierOutputSoftmaxTarget(combined_pred)]

            if label == specific_pred:
                specific_image_count += 1
                attributions = specific_cam_method(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
                specific_score = cam_metric(input_image, attributions, specific_metric_targets, specific_model)[0]
                scores[dataset_name][specific_model_name][label] += specific_score
                print('Score modelo específico:', specific_score)

            if label == combined_pred:
                combined_image_count += 1
                attributions = combined_cam_method(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
                combined_score = cam_metric(input_image, attributions, combined_metric_targets, combined_model)[0]
                scores[dataset_name][combined_model_name][label] += combined_score
                print('Score modelo combinado:', combined_score)

        print(dataset_name, 'scores before normalization:', scores[dataset_name])

        scores[dataset_name][specific_model_name] = {k: v / specific_image_count for k, v in scores[dataset_name][specific_model_name].items()}
        scores[dataset_name][combined_model_name] = {k: v / combined_image_count for k, v in scores[dataset_name][combined_model_name].items()}

        print(dataset_name, 'scores after normalization:', scores[dataset_name])

    with open('model_gradcam_metrics.pkl', 'wb') as f:
        pickle.dump(scores, f)

    return scores


def plot_metrics(metrics, savefig=None):
    dataset_names = list(metrics.keys())
    n_datasets = len(dataset_names)
    colors = {'specific1': 'darkgreen', 'combined1': '#66a266', 'specific2': 'darkblue', 'combined2': '#6666b9'}

    bar_width = 2
    num_bars = 4
    class_space = 0.5
    dataset_space = 2*bar_width
    index = np.arange(n_datasets) * (bar_width * num_bars + class_space + dataset_space)


    max_height = 0

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (dataset, models) in enumerate(metrics.items()):
        model_names = list(models.keys())
        class_labels = list(models[model_names[0]].keys())

        for j, cls in enumerate(class_labels):
            for k, model in enumerate(model_names):
                model_type = 'specific' if k == 0 else 'combined'
                label = f'Clase {cls}, Modelo {"específico" if k == 0 else "combinado"}' if i == 0 else None
                pos_x = index[i] + j * (2 + class_space) * bar_width + k * bar_width
                bars = ax.bar([pos_x], models[model][cls], bar_width, label=label, color=colors[model_type + str(cls)])
                for bar in bars:
                    height = bar.get_height()
                    max_height = max(max_height, height)
                    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                                textcoords="offset points", ha='center', va='bottom', fontsize=16)

    ax.tick_params(axis='x', which='both', direction='in', length=0, pad=15)
    ax.tick_params(axis='y', which='minor', direction='in', length=0, pad=15)
    ax.set_xlabel('Dataset', labelpad=10)
    ax.set_ylabel('ROADCombined promedio', labelpad=10)
    ax.set_ylim(0, max_height * 1.1)
    ax.set_title('ROADCombined promedio por clase y tipo de modelo', pad=20)
    ax.set_xticks([- bar_width / 2 + x + 2 * bar_width + class_space for x in index])
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=600)
    plt.show()


if __name__ == "__main__":
    #scores = compute_model_gradcam_scores()


    scores = {'ROB': {'resnet18_10_3_ROB': {1: 0.05951519504837368, 2: 0.08382157884214235}, 'resnet18_10_3_ROB_AO_HVV': {1: 0.05473481355742975, 2: 0.13626690005714243}}, 'AO': {'resnet18_10_3_ROB_AO': {1: 0.021376829594373703, 2: 0.1461890609934926}, 'resnet18_10_3_ROB_AO_HVV': {1: 0.034268077462911606, 2: 0.24899964220821857}}, 'HVV': {'resnet18_10_3_HVV': {1: 0.03938312758691609, 2: 0.08416067063808441}, 'resnet18_10_3_ROB_AO_HVV': {1: 0.11890941701437298, 2: 0.1013294305456312}}}

    print(scores)
    plot_metrics(metrics=scores, savefig='../figures/model_gradcam_scores.pdf')



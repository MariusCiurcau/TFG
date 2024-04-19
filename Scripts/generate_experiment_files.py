import csv
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from skimage.metrics import structural_similarity
from xplique.attributions import Saliency
from xplique.plots import plot_attributions
from xplique.plots.image import _clip_normalize
from xplique.wrappers import TorchWrapper

from Scripts.llm import generate_explanations_mistral
from Scripts.llm_gpt import generate_explanations_gpt
from Scripts.model import preprocess, preprocess_rgb
from utils import read_label, show_cam_on_image_alpha


specific_model_path = '../models/resnet18_10_3_ROB'
general_model_path = '../models/resnet18_10_3_ROB_AO_HVV'
img_dir = '../Datasets/COMBINED/resized_images'
label_dir = '../Datasets/COMBINED/augmented_labels'
num_classes = 3
classes_map = {0: 'No fracture', 1: 'Femoral neck fracture', 2: 'Trochanteric fracture'}
text_versions = ['Student', 'Expert']
SHOWN_VERSION = "Student"
USE_GPT = True


experiment_dir = '../experiment1'
experiment_imgs_dir = experiment_dir + '/images'
img_list_txt = experiment_imgs_dir + '/images.txt'
original_folder = experiment_imgs_dir + '/original'
specific_gradcam_folder = experiment_imgs_dir + '/specific_gradcam'
general_gradcam_folder = experiment_imgs_dir + '/general_gradcam'
xplique_folder = experiment_imgs_dir + '/xplique'
experiment_img_folders = [original_folder, specific_gradcam_folder, general_gradcam_folder, xplique_folder]
CSV_FILE = experiment_imgs_dir + '/images.csv'

if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)

with open(img_list_txt, 'r') as file:
    content = file.read()
images = content.split('\n')
images = [image.strip(';') for image in images if image]

images_dict = {}

for folder in experiment_img_folders:
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

for image in images:
    img_path = os.path.join(img_dir, image)
    image_name, _ = os.path.splitext(image)
    label_file = os.path.join(label_dir, image_name + '.txt')
    label = read_label(label_file, num_classes)
    images_dict[image] = label

specific_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
num_features = specific_model.fc.in_features
specific_model.fc = nn.Linear(num_features, num_classes)  # para resnet
specific_model.load_state_dict(torch.load(specific_model_path))
target_layers = [specific_model.layer4[-1]]  # especifico de resnet
specific_gradcam = GradCAM(specific_model, target_layers)  # Choose the last convolutional layer
specific_model.eval()

general_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
num_features = general_model.fc.in_features
general_model.fc = nn.Linear(num_features, num_classes)  # para resnet
general_model.load_state_dict(torch.load(general_model_path))
target_layers = [general_model.layer4[-1]]  # especifico de resnet
general_gradcam = GradCAM(general_model, target_layers)  # Choose the last convolutional layer
general_model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
xplique_model = TorchWrapper(general_model, device)
explainer = Saliency(xplique_model)

def gradcam(image_name, label, type):
    image = Image.open(os.path.join(img_dir, image_name))
    rgb_image = Image.open(os.path.join(img_dir, image_name))
    input_image = preprocess(image).unsqueeze(0)
    rgb_input_image = preprocess_rgb(rgb_image).permute(1, 2, 0).numpy()

    if type == 'specific':
        output = specific_model(input_image)
        pred = torch.argmax(output, 1)[0].item()
        attributions = specific_gradcam(input_tensor=input_image)
        attribution = attributions[0, :]
    elif type == 'general':
        output = general_model(input_image)
        pred = torch.argmax(output, 1)[0].item()
        attributions = general_gradcam(input_tensor=input_image)
        attribution = attributions[0, :]

    visualization = show_cam_on_image_alpha(rgb_input_image, attribution, use_rgb=True)
    return visualization, pred

def xplique(image_name, label):
    image = cv2.imread(os.path.join(img_dir, image_name))[..., ::-1]
    #image = Image.open(os.path.join(img_dir, image_name))
    X = np.array([image], dtype=np.uint8)
    Y = np.array([label])
    X_float = np.empty((1, 3, 224, 224), dtype=np.float32)
    for i, x in enumerate(X):
        X_float[i] = preprocess(x)
    X_preprocessed = torch.tensor(X_float, dtype=torch.float32)
    X_preprocessed4explainer = np.moveaxis(X_preprocessed.numpy(), [1, 2, 3], [3, 1, 2])
    batch_size = 1
    explanation = explainer(X_preprocessed4explainer, Y)

    attributions = _clip_normalize(explanation, clip_percentile=0.5, absolute_value=True)[0]
    attributions = (attributions * 255).astype(np.uint8)
    att_colormap = cv2.applyColorMap(attributions, cv2.COLORMAP_JET)
    #att_colormap_bgr = cv2.cvtColor(att_colormap, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image, 0.6, att_colormap, 0.4, 0)
    return overlay

    """
    input_image = preprocess(image)
    img_preprocessed = torch.tensor(np.array([input_image]), dtype=torch.float32)
    img_preprocessed = np.moveaxis(img_preprocessed.numpy(), [1, 2, 3], [3, 1, 2])
    batch_size = 1
    explanation = explainer(img_preprocessed, torch.tensor([label]))
    attribution = _clip_normalize(explanation, clip_percentile=0.5, absolute_value=True)
    return attribution
    """

def text_retrieval(image_name, label, use_gpt=True):
    img = cv2.imread(os.path.join(img_dir, image_name), cv2.IMREAD_GRAYSCALE)
    same_label_path = f'../Datasets/Dataset/Femurs/clusters/label{label}'
    same_label_dirs = [f.path for f in os.scandir(same_label_path) if f.is_dir() and f.name.startswith('cluster')]

    # same_cluster_path = f'../Datasets/Dataset/Femurs/clusters/label{pred}/cluster{cluster[0]}'
    dic_generalText = {0: 1, 1: 2, 2: 2}
    i = random.randint(1, dic_generalText[label])

    best_ssim = 0
    best_cluster = -1
    best_image_file = None
    for dir in same_label_dirs:
        current_cluster = int(dir.split('/')[-1][-1])
        for image_file in os.listdir(dir):
            print(image_file)
            if image_file.endswith(('.jpg', '.jpeg', '.png')) and not image_name.endswith(image_file):
                if not image_file.startswith(image_name):
                    img_aux = cv2.imread(dir + '/' + image_file, cv2.IMREAD_GRAYSCALE)
                    range_ = max(img.max() - img.min(), img_aux.max() - img_aux.min())
                    ssim = structural_similarity(img, img_aux, data_range=range_)
                    if ssim > best_ssim:
                        best_ssim = ssim
                        best_image_file = image_file
                        best_cluster = current_cluster

    best_image_name, _ = os.path.splitext(os.path.basename(best_image_file))
    print("Most similar image:", best_image_name)
    text_file_path = os.path.join(same_label_path, f'c{best_cluster}.txt')
    with open(text_file_path, 'r', encoding='utf-8') as text_file:
        texto = text_file.read()
    general_text_path = os.path.join(same_label_path, f'text{i}.txt')
    with open(general_text_path, 'r', encoding='utf-8') as text_file:
        general_text = text_file.read()

    if use_gpt:
        explanations = generate_explanations_gpt(texto, text_versions)
    else:
        explanations = generate_explanations_mistral(texto, text_versions)
    explanations = {version: general_text + '\n\n' + text for version, text in explanations.items()}
    explanation = explanations[SHOWN_VERSION].replace("\n", "")
    print('Explanation:', explanation)
    return explanation

def main():
    for image, label in images_dict.items():
        print(image, label)
        img = cv2.imread(os.path.join(img_dir, image))
        specific_overlay, specific_prediction = gradcam(image, label, 'specific')
        general_overlay, general_prediction = gradcam(image, label, 'general')
        xplique_overlay = xplique(image, label)

        pred = specific_prediction

        explanation = text_retrieval(image, pred, use_gpt=USE_GPT)

        with open(CSV_FILE, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow([image, classes_map[pred], explanation])

        cv2.imwrite(os.path.join(original_folder, image), img)
        cv2.imwrite(os.path.join(specific_gradcam_folder, image), specific_overlay)
        cv2.imwrite(os.path.join(general_gradcam_folder, image), general_overlay)
        cv2.imwrite(os.path.join(xplique_folder, image), xplique_overlay)

if __name__ == '__main__':
    main()
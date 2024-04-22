import csv
import os
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from skimage.metrics import structural_similarity
from xplique.attributions import Saliency
from xplique.plots.image import _clip_normalize
from xplique.wrappers import TorchWrapper

from Scripts.llm import generate_explanations_mistral
from Scripts.llm_gpt import generate_explanations_gpt
from Scripts.model import preprocess, preprocess_rgb
from utils import read_label, show_cam_on_image_alpha

two_class_model_path = '../models/resnet18_10_2_ROB_AO_HVV'
three_class_model_path = '../models/resnet18_10_3_ROB_AO_HVV'

num_classes = 3
classes_map = {0: 'No fracture', 1: 'Femoral neck fracture', 2: 'Trochanteric fracture'}
text_versions = ['Student', 'Expert']
SHOWN_VERSION = "Student"
USE_GPT = True

random.seed(0)

def read_images_list(images_list, datasets_path, classes):
    images_dict = {}
    sources_dict = {}

    with open(images_list, "r") as file:
        content = file.read()
    images_list = [image.strip() for image in content.split(";")][:-1]

    for image in images_list:
        image_name, _ = os.path.splitext(image)
        source = image_name.split("_")[0]
        label_file = os.path.join(datasets_path, source, 'labels', image_name + '.txt')
        label = read_label(label_file, num_classes)
        if label in classes:
            images_dict[image] = label
            sources_dict[image] = datasets_path + '/' + source
        else:
            print("Skipping", image)

    return images_dict, sources_dict

def generate_image_list(sources, classes):
    images_dict = {}
    sources_dict = {}
    for source, images_per_source in sources.items():
        img_dir = os.path.join(source, 'images')
        label_dir = os.path.join(source, 'labels')
        imgs = os.listdir(img_dir)
        random.shuffle(imgs)
        i = 0
        fractures_found = 0
        while i < len(imgs) and fractures_found < images_per_source:
            image = imgs[i]
            image_name, _ = os.path.splitext(image)
            label_file = os.path.join(label_dir, image_name + '.txt')
            label = read_label(label_file, num_classes)
            if label in classes:
                images_dict[image] = label
                sources_dict[image] = source
                fractures_found += 1
            i += 1
    return images_dict, sources_dict


two_class_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
num_features = two_class_model.fc.in_features
two_class_model.fc = nn.Linear(num_features, 2)  # para resnet
two_class_model.load_state_dict(torch.load(two_class_model_path))
target_layers = [two_class_model.layer4[-1]]
two_class_gradcam = GradCAM(two_class_model, target_layers)  # Choose the last convolutional layer
two_class_model.eval()

three_class_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
num_features = three_class_model.fc.in_features
three_class_model.fc = nn.Linear(num_features, num_classes)  # para resnet
three_class_model.load_state_dict(torch.load(three_class_model_path))
target_layers = [three_class_model.layer4[-1]]  # etwo_classo de resnet
three_class_gradcam = GradCAM(three_class_model, target_layers)  # Choose the last convolutional layer
three_class_model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
xplique_model = TorchWrapper(three_class_model, device)
explainer = Saliency(xplique_model)


def gradcam(image_path, label, type):
    image = Image.open(image_path)
    rgb_image = Image.open(image_path)
    input_image = preprocess(image).unsqueeze(0)
    rgb_input_image = preprocess_rgb(rgb_image).permute(1, 2, 0).numpy()

    if type == 'two_class':
        output = two_class_model(input_image)
        pred = torch.argmax(output, 1)[0].item()
        attributions = two_class_gradcam(input_tensor=input_image)
        attribution = attributions[0, :]
    elif type == 'three_class':
        output = three_class_model(input_image)
        pred = torch.argmax(output, 1)[0].item()
        attributions = three_class_gradcam(input_tensor=input_image)
        attribution = attributions[0, :]

    visualization = show_cam_on_image_alpha(rgb_input_image, attribution, image_weight=0.175, use_rgb=True)
    return visualization, pred


def xplique(image_path, label):
    image = cv2.imread(image_path)[..., ::-1]
    # image = Image.open(os.path.join(img_dir, image_name))
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
    # att_colormap_bgr = cv2.cvtColor(att_colormap, cv2.COLOR_RGB2BGR)
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


def text_retrieval(image_path, label, use_gpt=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_name, _ = os.path.splitext(os.path.basename(image_path))
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
            if (image_file.endswith(('.jpg', '.jpeg', '.png'))
                    and not image_path.endswith(image_file)
                    and not image_file.startswith(image_name)):
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
        explanations = generate_explanations_gpt(texto, [SHOWN_VERSION])
    else:
        explanations = generate_explanations_mistral(texto, [SHOWN_VERSION])
    explanations = {version: general_text + ' ' + text for version, text in explanations.items()}
    explanation = explanations[SHOWN_VERSION].replace("\n", "")
    print('Explanation:', explanation)
    return explanation


def generate_experiment1(classes):
    experiment_dir = '../experiment1'
    experiment_imgs_dir = experiment_dir + '/images'
    img_list_txt = experiment_imgs_dir + '/images.txt'
    original_folder = experiment_imgs_dir + '/original'
    two_class_gradcam_folder = experiment_imgs_dir + '/two_class_gradcam'
    three_class_gradcam_folder = experiment_imgs_dir + '/three_class_gradcam'
    xplique_folder = experiment_imgs_dir + '/xplique'
    experiment_img_folders = [original_folder, two_class_gradcam_folder, three_class_gradcam_folder, xplique_folder]
    CSV_FILE = experiment_imgs_dir + '/images.csv'
    datasets_path = '../Datasets'

    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    labels_dict, sources_dict = read_images_list(img_list_txt, datasets_path, classes)

    for folder in experiment_img_folders:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    for image, label in labels_dict.items():
        img_dir = os.path.join(sources_dict[image], 'images')
        img = cv2.imread(os.path.join(img_dir, image))
        two_class_overlay, two_class_prediction = gradcam(os.path.join(img_dir, image), label, 'two_class')
        three_class_overlay, three_class_prediction = gradcam(os.path.join(img_dir, image), label, 'three_class')
        xplique_overlay = xplique(os.path.join(img_dir, image), label)

        pred = three_class_prediction

        with open(CSV_FILE, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow([image, classes_map[pred]])

        cv2.imwrite(os.path.join(original_folder, image), img)
        cv2.imwrite(os.path.join(two_class_gradcam_folder, image), two_class_overlay)
        cv2.imwrite(os.path.join(three_class_gradcam_folder, image), three_class_overlay)
        cv2.imwrite(os.path.join(xplique_folder, image), xplique_overlay)


def generate_experiment2(sources, classes):
    experiment_dir = '../experiment2'
    experiment_imgs_dir = experiment_dir + '/images'
    img_list_txt = experiment_imgs_dir + '/images.txt'
    original_folder = experiment_imgs_dir + '/original'
    experiment_img_folders = [original_folder]
    CSV_FILE = experiment_imgs_dir + '/images.csv'

    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    if os.path.exists(img_list_txt):
        os.remove(img_list_txt)

    labels_dict, sources_dict = generate_image_list(sources, classes=classes)

    for folder in experiment_img_folders:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    for image, label in labels_dict.items():
        img_dir = os.path.join(sources_dict[image], 'images')
        img = cv2.imread(os.path.join(img_dir, image))
        three_class_overlay, three_class_prediction = gradcam(os.path.join(img_dir, image), label, 'three_class')
        pred = three_class_prediction

        explanation = text_retrieval(os.path.join(img_dir, image), pred, use_gpt=USE_GPT)

        with open(CSV_FILE, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow([image, classes_map[pred], explanation])

        with open(img_list_txt, 'a', newline='') as file:
            file.write(f'{image};\n')

        cv2.imwrite(os.path.join(original_folder, image), img)


def generate_experiment3(sources, classes):
    experiment_dir = '../experiment3'
    experiment_imgs_dir = experiment_dir + '/images'
    img_list_txt = experiment_imgs_dir + '/images.txt'
    original_folder = experiment_imgs_dir + '/original'
    experiment_img_folders = [original_folder]
    CSV_FILE = experiment_imgs_dir + '/images.csv'

    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    if os.path.exists(img_list_txt):
        os.remove(img_list_txt)

    labels_dict, sources_dict = generate_image_list(sources, classes=classes)

    for folder in experiment_img_folders:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    for image, label in labels_dict.items():
        img_dir = os.path.join(sources_dict[image], 'images')
        img = cv2.imread(os.path.join(img_dir, image))
        # two_class_overlay, two_class_prediction = gradcam(os.path.join(img_dir, image), label, 'two_class')
        three_class_overlay, three_class_prediction = gradcam(os.path.join(img_dir, image), label, 'three_class')

        pred = three_class_prediction

        with open(CSV_FILE, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow([image, label, pred])

        with open(img_list_txt, 'a', newline='') as file:
            file.write(f'{image};\n')

        cv2.imwrite(os.path.join(original_folder, image), img)


if __name__ == '__main__':
    generate_experiment1(classes=[1, 2])
    #generate_experiment2(sources={'../Datasets/ROB': 5, '../Datasets/AO': 5, '../Datasets/HVV': 5}, classes=[1, 2])
    # generate_experiment3(sources={'../Datasets/ULTIMAS': 50}, classes=[0, 1, 2])

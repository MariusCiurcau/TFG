import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_grad_cam.metrics.road import ROADCombined
from skimage.metrics import structural_similarity
from sklearn.model_selection import train_test_split, KFold

import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, SubsetRandomSampler, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, AblationCAM, EigenGradCAM, RandomCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from llm import generate_explanations_mistral
from llm_gpt import generate_explanations_gpt
from utils import find_similar_images, visualize_label, add_border, read_label, add_filename

from xplique.attributions import Rise
from xplique.metrics import Deletion
from xplique.plots import plot_attributions
from xplique.wrappers import TorchWrapper
import re
from gui import show_gui

import scienceplots
plt.style.use(['science', 'no-latex'])

USE_GPT = False

torch.manual_seed(0)

def evaluate_model(model, criterion, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    corrects = 0
    samples = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_loss += criterion(outputs, labels).item()
            corrects += torch.sum(torch.argmax(outputs, 1) == labels)
            samples += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = corrects / samples
    loss = total_loss / samples

    return accuracy, loss

def get_predictions(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


def calculate_accuracy(outputs, labels):
    accuracy = torch.sum(outputs == labels).item() / len(labels)
    return accuracy


preprocess = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_rgb = transforms.Compose([
    transforms.ToTensor(),
])


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_eval_model(df, epochs=None, split=None, sample=None, save_path=None, load_path=None, crossval=False, num_classes=3):
    val_split = False
    if split is None:
        split = [0.8, 0.2]

    if len(split) == 3:
        val_split = True

    X_aux, X_test, y_aux, y_test = train_test_split(
        df[['filename', 'data']], df.label.values, test_size=split[1], shuffle=True, random_state=1, stratify=df.label.values
    )
    X_test = np.array([item for item in X_test.data.values])

    if val_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X_aux, y_aux, test_size=split[2] / (1 - split[1]), shuffle=True, random_state=1, stratify=y_aux
        )
        X_val = np.array([item for item in X_val.data.values])
    else:
        X_train, y_train = X_aux, y_aux


    if sample is not None:
        X_train.reset_index(drop=True, inplace=True)
        y_train_df = pd.DataFrame({'label': y_train})
        df_train = pd.concat([X_train, y_train_df], axis=1)
        df_original = df_train[df_train['filename'].str.endswith('_0.jpg')]

        unique, counts = np.unique(df_original['label'], return_counts=True)
        print("Sample original:\n", np.asarray((unique, counts)).T)
        
        df_augmented = df_train[~df_train['filename'].str.endswith('_0.jpg')]
        df_sample = pd.DataFrame(columns=df_train.columns)
        for label, count in sample.items():
            df_label_original = df_original[df_original['label'] == label]
            df_label_sample = df_label_original.sample(min(len(df_label_original), count), random_state=42)
            df_sample = pd.concat([df_sample, df_label_sample], ignore_index=True)
            n_augmentations = max(0, count - len(df_label_original))
            if n_augmentations > 0:
                df_label_augmented = df_augmented[df_augmented['label'] == label]
                df_label_sample = df_label_augmented.sample(min(n_augmentations, len(df_label_augmented)), random_state=42)
                df_sample = pd.concat([df_sample, df_label_sample], ignore_index=True)
        X_train = np.array([item for item in df_sample.data.values], dtype=np.uint8)
        y_train = np.array(df_sample['label'].values.astype('int'))
    else:
        X_train = np.array([item for item in X_train.data.values], dtype=np.uint8)

    unique, counts = np.unique(y_train, return_counts=True)
    print("Sample:\n", np.asarray((unique, counts)).T)

    X_train_float = np.empty((len(X_train), 224, 224, 3), dtype=np.float32)
    for i, img in enumerate(X_train):
        X_train_float[i] = preprocess(img).permute(1, 2, 0)
    X_train = X_train_float

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_float = np.empty((len(X_test), 224, 224, 3), dtype=np.float32)
    for i, img in enumerate(X_test):
        X_test_float[i] = preprocess(img).permute(1, 2, 0)
    X_test = X_test_float

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Crear conjuntos de datos y DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    if crossval:
        k_folds = 5
        dataset = ConcatDataset([train_dataset, test_dataset])
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        accuracies = {}
        losses = {}
        last_accuracy = []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            accuracies['Fold' + str(fold+1)] = {'train': [], 'test': []}
            losses['Fold' + str(fold+1)] = {'train': [], 'test': []}

            print(f'Fold {fold+1}/{k_folds}')
            print('--------------------------------')
            train_sampler = SubsetRandomSampler(train_ids)
            test_sampler = SubsetRandomSampler(test_ids)
            train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler, shuffle=False)
            test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler, shuffle=False)

            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            model.apply(reset_weights)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # para resnet
            best_test_accuracy = 0
            if epochs is not None:
                num_epochs = epochs
            else:
                num_epochs = 10
            for epoch in range(num_epochs):
                print("Epoch {}/{}".format(epoch + 1, num_epochs))
                model.train()
                train_corrects = 0
                train_samples = 0
                total_loss = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    train_corrects += torch.sum(torch.argmax(outputs, 1) == labels)
                    train_samples += len(labels)
                train_accuracy = train_corrects / train_samples
                train_loss = total_loss / train_samples

                test_accuracy, test_loss = evaluate_model(model, criterion, test_loader)
                best_test_accuracy = max(test_accuracy, best_test_accuracy)

                print(f"\tTrain Accuracy: {train_accuracy:.6f}, Loss: {train_loss:.6f}")
                print(f"\tTest Accuracy: {test_accuracy:.6f}, Loss: {test_loss:.6f}")

                last_accuracy.append(test_accuracy)

                accuracies['Fold' + str(fold+1)]['train'].append(train_accuracy)
                accuracies['Fold' + str(fold+1)]['test'].append(test_accuracy)
                losses['Fold' + str(fold+1)]['train'].append(train_loss)
                losses['Fold' + str(fold+1)]['test'].append(test_loss)

        print(f"Average test accuracy: {sum(last_accuracy) / len(last_accuracy):.6f}")

        for fold, data in accuracies.items():
            test_values = data['test']
            plt.plot(range(1, len(test_values) + 1), test_values, label=f'Fold {fold[-1]}')

        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(title="Fold")
        plt.savefig('../figures/test_accuracy.png', dpi=600)
        plt.show()

        for fold, data in losses.items():
            test_values = data['test']
            plt.plot(range(1, len(test_values) + 1), test_values, label=f'Fold {fold[-1]}')

        plt.title('Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(title="Fold")
        plt.savefig('../figures/test_loss.png', dpi=600)
        plt.show()


    else:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)  # para resnet
        if load_path is not None:
            model.load_state_dict(torch.load(load_path))
        else:
            writer = SummaryWriter()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # para resnet
            if epochs is not None:
                num_epochs = epochs
            else:
                num_epochs = 10
            for epoch in range(num_epochs):
                print("Epoch {}/{}".format(epoch + 1, num_epochs))
                model.train()
                train_corrects = 0
                train_samples = 0
                total_loss = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    train_corrects += torch.sum(torch.argmax(outputs, 1) == labels)
                    train_samples += len(labels)
                train_accuracy = train_corrects / train_samples
                train_loss = total_loss / train_samples

                if (epoch + 1) % 5 == 0 and (epoch + 1) < num_epochs:
                    torch.save(model.state_dict(), save_path + '_epoch' + str(epoch + 1))

                test_accuracy, test_loss = evaluate_model(model, criterion, test_loader)

                print(f"\tTrain Accuracy: {train_accuracy:.6f}, Loss: {train_loss:.6f}")
                print(f"\tTest Accuracy: {test_accuracy:.6f}, Loss: {test_loss:.6f}")

                writer.add_scalar('Accuracy/train', train_accuracy, epoch + 1)
                writer.add_scalar('Accuracy/test', test_accuracy, epoch + 1)
                writer.add_scalar('Loss/train', train_loss, epoch + 1)
                writer.add_scalar('Loss/test', test_loss, epoch + 1)
            writer.flush()
            writer.close()

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    predicted_labels, true_labels = get_predictions(model, test_loader)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    report = classification_report(true_labels, predicted_labels)
    print(report)
    return report, str(conf_matrix)


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, num_classes, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [image for image in os.listdir(self.image_dir) if image.endswith(('.jpg', '.jpeg', '.png'))]
        # Load labels from text files
        self.labels = self._load_labels(num_classes=num_classes)

    def _load_labels(self, num_classes):
        labels = []
        for image_file in self.image_files:
            label_file = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + '.txt')
            label = read_label(label_file, num_classes)
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


def predict(load_path, image_path=None, labels_path=None, num_classes=3):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # para resnet
    model.load_state_dict(torch.load(load_path))
    target_layers = [model.layer4[-1]]  # especifico de resnet
    gradcam = GradCAM(model, target_layers)  # Choose the last convolutional layer
    model.eval()

    image_dir = '../Datasets/COMBINED/resized_images'
    label_dir = '../Datasets/COMBINED/augmented_labels'

    if image_path is not None:
        if os.path.isdir(image_path) and labels_path is not None:
            dataset = CustomDataset(image_dir=image_path, label_dir=labels_path, num_classes=num_classes, transform=preprocess)
            predict_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            predicted_labels, true_labels = get_predictions(model, predict_loader)
            conf_matrix = confusion_matrix(true_labels, predicted_labels)

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(true_labels),
                        yticklabels=np.unique(true_labels))
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            #plt.show()

            report = classification_report(true_labels, predicted_labels)
            print(report)
            return report, str(conf_matrix)
        elif os.path.isdir(image_path) and labels_path is None:
            print('No labels path provided')
            exit(1)
        else:
            image = Image.open(image_path)
            image_name, _ = os.path.splitext(os.path.basename(image_path))
            label_file = os.path.join(label_dir, image_name + '.txt')
            if(os.path.exists(label_file)):
                label = read_label(label_file, num_classes)
            else:
                label = None
            rgb_image = Image.open(image_path)

            input_image = preprocess(image).unsqueeze(0)
            rgb_input_image = preprocess_rgb(rgb_image).permute(1, 2, 0).numpy()

            attributions = gradcam(input_tensor=input_image)
            attribution = attributions[0, :]
            output = model(input_image)
            pred = torch.argmax(output, 1)[0].item()

            if label != 0:
                visualization = show_cam_on_image(rgb_input_image, attribution, use_rgb=True)
            else:
                visualization = np.array(image)
            visualization = visualize_label(visualization, label, pred)
            visualization = add_border(visualization, label, pred)

            class FeatureExtractor(nn.Module):
                def __init__(self, model):
                    super(FeatureExtractor, self).__init__()
                    self.features = nn.Sequential(
                        *list(model.children())[:-2],  # Remove avgpool and fc layers
                        nn.AdaptiveAvgPool2d((1, 1))
                    )

                def forward(self, x):
                    return self.features(x)

            model = FeatureExtractor(model)
            model.eval()

            #model.fc = nn.Identity()  # para clutering por features

            # clustering por features
            #features = model(input_image).detach().numpy()[0].flatten().astype(np.double)
            #images_predict = np.array([features])

            # clustering sin features
            images_predict = np.array([np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)).flatten()])


            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            kmeans_list = np.array([])
            kmeans_list = np.append(kmeans_list,pickle.load(open("../clusters/clusterClase1.pkl", "rb")))
            kmeans_list = np.append(kmeans_list,pickle.load(open("../clusters/clusterClase2.pkl", "rb")))
            #el indice es label-1 porque no hay KMEANS para la clase 0
            kmeans = kmeans_list[pred-1]

            cluster = kmeans.predict(images_predict)

            #Search for similar + text
            same_label_path = f'../Datasets/Dataset/Femurs/clusters/label{pred}'
            same_cluster_path = f'../Datasets/Dataset/Femurs/clusters/label{pred}/cluster{cluster[0]}'
            dic_generalText = {0:1, 1:2, 2:2}
            i = random.randint(1, dic_generalText[pred])

            best_ssim = 0
            for image_file in os.listdir(same_cluster_path):
                print(image_file)
                if image_file.endswith(('.jpg','.jpeg','.png')) and not image_path.endswith(image_file):
                        if not image_file.startswith(image_name):
                            print(image_file)
                            img_aux = cv2.imread(same_cluster_path + '/' + image_file, cv2.IMREAD_GRAYSCALE)
                            range_ = max(img.max() - img.min(), img_aux.max() - img_aux.min())
                            ssim = structural_similarity(img, img_aux, data_range=range_)
                            if ssim > best_ssim:
                                best_ssim = ssim
                                best_image_file = image_file
                        else:
                            print(image_file)

            best_image_name, _ = os.path.splitext(os.path.basename(best_image_file))
            print("Most similar image:", best_image_name)
            text_file_path = os.path.join(same_label_path, f'c{cluster[0]}.txt')
            with open(text_file_path, 'r', encoding='utf-8') as text_file:
                texto = text_file.read()
            general_text_path = os.path.join(same_label_path, f'text{i}.txt')
            with open(general_text_path, 'r', encoding='utf-8') as text_file:
                general_text = text_file.read()

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 6))

            plt.tight_layout(pad=2.0)

            fig.suptitle('Image comparison', fontsize=14)

            # Título de las imágenes
            axes[0, 0].set_title('Explanation')
            axes[0, 1].set_title('Original image')
            axes[0, 2].set_title('Most similar image')
            axes[1, 1].set_title('General Diagnosis')
            axes[2, 1].set_title('Particular Diagnosis')

            # Mostrar las imágenes y el texto
            axes[0, 0].imshow(visualization, cmap='gray')
            axes[0, 0].axis('off')
            axes[0, 1].imshow(img, cmap='gray')
            axes[0, 1].axis('off')
            axes[0, 2].imshow(Image.open(same_cluster_path + '/' + best_image_file), cmap='gray')
            axes[0, 2].axis('off')
            axes[1, 1].text(0.5, 0.5, s=general_text, ha='center', va='center', fontsize=10, wrap=True) # Ajuste para envolver el texto
            axes[1, 1].axis('off')
            axes[1,0].axis('off')
            axes[1,2].axis('off')
            axes[2,0].axis('off')
            axes[2, 1].text(0.5, 0.5, s=texto, ha='center', va='center', fontsize=10, wrap=True) # Ajuste para envolver el texto
            axes[2,1].axis('off')
            axes[2,2].axis('off')

            # Ajustar los tamaños de los subgráficos y la distancia vertical entre ellos
            plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, hspace=0.5)  # Ajustar la distancia vertical entre los subgráficos

            # Mostrar la figura
            plt.show()

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img)
            visualization = Image.fromarray(visualization)
            versions = ['Student', 'Expert']
            explanations_mistral = generate_explanations_mistral(texto, versions)
            explanations_mistral = {version: general_text + '\n\n' + text for version, text in explanations_mistral.items()} # we add the general text
            if USE_GPT:
                explanations_gpt = generate_explanations_gpt(texto, versions)
            else:
                explanations_gpt = {version: 'Text not available.' for version in versions}
            explanations_gpt = {version: general_text + '\n\n' + text for version, text in
                                explanations_gpt.items()}  # we add the general text
            explanations = {'Mistral': explanations_mistral, 'GPT4': explanations_gpt}
            show_gui({'Explanation': visualization, 'Original image': img, 'Most similar image': Image.open(same_cluster_path + '/' + best_image_file)}, explanations)
    else:
        image_files = os.listdir(image_dir)
        image_files = [image_file for image_file in image_files if image_file.endswith('_0.jpg')]
        random.shuffle(image_files)

        visualizations = []
        images = []
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
            cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
            metric_targets = [ClassifierOutputSoftmaxTarget(pred)]

            methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers)),
                       ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers)),
                       ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers)),
                       ("AblationCAM", AblationCAM(model=model, target_layers=target_layers)),
                       ("RandomCAM", RandomCAM(model=model, target_layers=target_layers))]

            visualizations_aux = []
            if label != 0:
                for name, cam_method in methods:
                    attributions = cam_method(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
                    attribution = attributions[0, :]
                    scores = cam_metric(input_image, attributions, metric_targets, model)
                    score = scores[0]
                    visualization = show_cam_on_image(rgb_input_image, attribution, use_rgb=True)
                    visualization = visualize_label(visualization, label, pred, name=name, score=score)
                    visualization = add_border(visualization, label, pred)
                    visualizations_aux.append(visualization)
            else:
                similar_images = find_similar_images(image_path, label, image_files, image_dir, label_dir, num_images=5, num_classes=num_classes)
                for similar_image in similar_images:
                    visualization = np.array(cv2.imread(image_dir + '/' + similar_image))
                    visualization = visualize_label(visualization, label, pred, similar=True)
                    visualization = add_border(visualization, label, pred)
                    visualizations_aux.append(visualization)

            images.append(add_filename(np.array(image), image_name[:-2])) # we add the image name
            visualizations.append(visualizations_aux)

        fig, axes = plt.subplots(1 + len(methods), 5, figsize=(10, 2 * (1 + len(methods))))
        plt.subplots_adjust(wspace=0, hspace=0)

        for i in range(5):
            axes[0, i].imshow(images[i], cmap='gray')
            axes[0, i].axis('off')

        for i in range(5):
            for j in range(len(methods)):
                axes[j + 1, i].imshow(visualizations[i][j])
                axes[j + 1, i].axis('off')

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or infer a model.')

    # Add arguments for training and inference
    parser.add_argument('--train', action='store_true', help='Train and evaluate the model')
    parser.add_argument('--predict', action='store_true', help='Perform inference using the trained model')
    parser.add_argument('--load', type=str, help='Path to a pre-trained model')
    parser.add_argument('--save', type=str, help='Path where model will be saved')
    parser.add_argument('--image', type=str, help='Path to input image/s')
    parser.add_argument('--labels', type=str, help='Path to labels')
    parser.add_argument('--num_classes', type=int, help='Number of classes')

    args = parser.parse_args()

    if args.train:
        load_path = None
        save_path = None
        if args.load:
            load_path = args.load
        if args.save:
            save_path = args.save
        if args.rgb:
            df = pd.read_pickle("../df_rgb.pkl")
        else:
            df = pd.read_pickle("../df.pkl")
        if args.num_classes:
            num_classes = args.num_classes
        else:
            num_classes = 3
        train_eval_model(df, epochs=10, split=[0.8, 0.2], sample={0: 1000, 1: 1000}, load_path=load_path, save_path=save_path, num_classes=num_classes)
    elif args.predict:
        load_path = None
        if args.load is None:
            parser.error("--load is required when performing inference.")
        if args.num_classes:
            num_classes = args.num_classes
        else:
            num_classes = 2

        predict(load_path=args.load, image_path=args.image, labels_path=args.labels, num_classes=num_classes)
    else:
        print("Please provide either --train or --predict argument.")
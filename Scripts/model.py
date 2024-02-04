import argparse
import math
import os
import random
from typing import Callable, Union
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_grad_cam.metrics.road import ROADCombined
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn.modules.module import _grad_t
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import cv2
from torch.utils.hooks import RemovableHandle
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, \
    EigenGradCAM, RandomCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

torch.manual_seed(0)

def visualize_label(visualization, label, prediction):
    visualization = cv2.putText(visualization, f"Class: {label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Prediction: {prediction}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return visualization

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size, in_channels):
        super(ConvNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=(3, 3), padding="same"),
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

# GradCAM implementation
class GradCAM2:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activations = None

        # Register the hook and keep the handle for later removal
        self.hook_handle = self.hook()

    def hook(self):
        def hook_fn(module, input, output):
            self.activations = output
            self.activations.retain_grad()
        hook_handle = self.target_layer.register_forward_hook(hook_fn)
        return hook_handle

    def compute_gradient(self, input_image, target_class=None):
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_image)

        # Backward pass
        if target_class is None:
            target_class = torch.argmax(output)
        loss = output[0, target_class]
        loss.backward()

        # Detach the activations to avoid the warning
        self.gradient = self.activations.grad

    def generate_heatmap(self):
        if self.gradient is None:
            raise ValueError("Gradients are not computed. Call compute_gradient first.")

        weights = torch.mean(self.gradient, dim=(1, 2), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        heatmap = nn.functional.relu(heatmap)

        return heatmap.squeeze()  # Squeeze the dimensions to remove the singleton dimensions

    def remove_hook(self):
        self.hook_handle.remove()

def backward_hook(module, grad_input, grad_output):
  global gradients # refers to the variable in the global scope
  print('Backward hook running...')
  gradients = grad_output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Gradients size: {gradients[0].size()}')
  # We need the 0 index because the tensor containing the gradients comes
  # inside a one element tuple.

def forward_hook(module, args, output):
  global activations # refers to the variable in the global scope
  print('Forward hook running...')
  activations = output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Activations size: {activations.size()}')

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
    #transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_rgb = transforms.Compose([
    #transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def train_eval_model(df, epochs=None, split=None, sample=None, save_path=None, load_path=None, rgb=False):
    val_split = False
    if split is None:
        split = [0.8, 0.2]

    if len(split) == 3:
        val_split = True

    #df['data'] = df['data'].apply(lambda x: x.flatten())
    #flattened_data = np.array([item.flatten() for item in df.data.values])
    X_aux, X_test, y_aux, y_test = train_test_split(
        df[['filename', 'data']], df.label.values, test_size=split[1], shuffle=True, random_state=1, stratify=df.label.values
    )
    #X_test = np.array([item.flatten() for item in X_test.data.values])
    X_test = np.array([item for item in X_test.data.values])

    if val_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X_aux, y_aux, test_size=split[2] / (1 - split[1]), shuffle=True, random_state=1, stratify=y_aux
        )
        # X_val = np.array([item.flatten() for item in X_val.data.values])
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
        #X_train = np.array([item.flatten() for item in df_sample.data.values])
        X_train = np.array([item for item in df_sample.data.values], dtype=np.uint8)
        y_train = np.array(df_sample['label'].values.astype('int'))
    else:
        #X_train = np.array([item.flatten() for item in X_train.data.values])
        X_train = np.array([item for item in X_train.data.values], dtype=np.uint8)

    unique, counts = np.unique(y_train, return_counts=True)
    print("Sample:\n", np.asarray((unique, counts)).T)

    X_train_float = np.empty((len(X_train), 224, 224, 3), dtype=np.float32)
    for i, img in enumerate(X_train):
        """
        if i == 0:
            print("Imagen sin procesar:", img)
            plt.imshow(img)
            plt.show()

            print("Imagen procesada:", preprocess(img).permute(1, 2, 0))
            plt.imshow(preprocess(img).permute(1, 2, 0))
            plt.show()
        

        img_aux = cv2.Canny(img, 100, 200)
        img_aux = cv2.cvtColor(img_aux, cv2.COLOR_GRAY2RGB)
        if i == 0:
            plt.imshow(img)
            plt.show()

            plt.imshow(preprocess(img_aux).permute(1, 2, 0))
            plt.show()


        #print(type(X_train[i]))
        print(type(preprocess(img_aux).permute(1, 2, 0)))
        """
        X_train_float[i] = preprocess(img).permute(1, 2, 0)
        #if i == 0:
        #    print("Imagen procesada:", X_train[i])
        #    plt.imshow(X_train[i]*255.0)
        #    plt.show()
    X_train = X_train_float

    #X_train_tensor = preprocess(X_train)  # para resnet
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    if rgb:
        X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
    else:
        X_train_tensor = X_train_tensor.unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_float = np.empty((len(X_test), 224, 224, 3), dtype=np.float32)
    for i, img in enumerate(X_test):
        #img_aux = cv2.Canny(img, 100, 200)
        #img_aux = cv2.cvtColor(img_aux, cv2.COLOR_GRAY2RGB)
        #if i == 125:
        #    plt.imshow(img_aux)
        #    plt.show()
        #X_test[i] = preprocess(img_aux).permute(1, 2, 0)

        """
        if i == 1:
            img = img / 255.0
            sigma1 = 0.3003866304138461 * (9.0 + 1.0)
            sigma2 = 0.3003866304138461 * (11.0 + 1.0)
            gaussian_blur1 = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
            gaussian_blur2 = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
            dog = gaussian_blur2 - gaussian_blur1
            plt.imshow(dog*255.0)
            plt.show()
        """

        X_test_float[i] = preprocess(img).permute(1, 2, 0)
    X_test = X_test_float

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    if rgb:
        X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)
    else:
        X_test_tensor = X_test_tensor.unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Crear conjuntos de datos y DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1] * X_train.shape[2]
    channels = 1 if len(X_train.shape) < 4 else X_train.shape[3]
    output_size = 2  # Ajusta esto según el número de clases en tu problema
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    model.fc = nn.Linear(512, 2)  # para resnet
    #model = ConvNet(input_size, output_size, channels)

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    else:
        writer = SummaryWriter()
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=0.00005)
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

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), save_path + '_epoch' + str(epoch + 1))

            # Evaluate on validation set
            #test_preds, test_labels = evaluate_model(model, test_loader)

            # Calculate test accuracy and loss
            test_accuracy, test_loss = evaluate_model(model, criterion, test_loader)

            #test_loss = criterion(torch.tensor(test_preds).float(), torch.tensor(test_labels).float())

            print(f"\tTrain Accuracy: {train_accuracy:.6f}, Loss: {train_loss:.6f}")
            print(f"\tTest Accuracy: {test_accuracy:.6f}, Loss: {test_loss:.6f}")

            # Log training accuracy
            writer.add_scalar('Accuracy/train', train_accuracy, epoch + 1)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch + 1)
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Loss/test', test_loss, epoch + 1)
        writer.flush()
        writer.close()

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    # Evaluar el modelo en el conjunto de prueba
    predicted_labels, true_labels = get_predictions(model, test_loader)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1],
                yticklabels=[0, 1])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    report = classification_report(true_labels, predicted_labels)

    # Imprimir el reporte de clasificación
    print(report)

    """
    names = [
        # "Linear SVM",
        # "RBF SVM",
        # "Decision Tree",
        "Neural Net"
    ]

    classifiers = [
        # SVC(kernel="linear", C=0.025, random_state=42),
        # SVC(gamma=2, C=1, random_state=42),
        # DecisionTreeClassifier(max_depth=5, random_state=42),
        MLPClassifier(random_state=42, max_iter=300, early_stopping=True)
    ]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = f"Classification report for classifier {clf}:\n{metrics.classification_report(y_test, predicted)}"
        print(report)
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix for " + name)
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.show()
    conf_mat = disp.confusion_matrix
    """

    return report, str(conf_matrix)


def predict(load_path, width, height, image_path=None, rgb=False):
    #model = ConvNet(width * height, 2,in_channels= 3 if rgb else 1)
    #model.load_state_dict(torch.load(load_path))
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    model.fc = nn.Linear(512, 2) # para resnet
    model.load_state_dict(torch.load(load_path))
    target_layers = [model.layer4[-1]] # especifico de resnet
    gradcam = GradCAM(model, target_layers)  # Choose the last convolutional layer
    model.eval()

    image_dir = '../Datasets/Dataset/Femurs/resized_images'
    label_dir = '../Datasets/Dataset/Femurs/augmented_labels_fractura'

    if image_path is not None:
        if rgb:
            image = Image.open(image_path)
        else:
            image = Image.open(image_path).convert("L")
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        label_file = os.path.join(label_dir, image_name + '.txt')
        with open(label_file, 'r') as file:
            label = file.read()
        rgb_image = Image.open(image_path)

        #img_aux = cv2.Canny(image, 100, 200)
        #img_aux = cv2.cvtColor(img_aux, cv2.COLOR_GRAY2RGB)
        #input_image = preprocess(img_aux).permute(1, 2, 0).unsqueeze(0)

        input_image = preprocess(image).unsqueeze(0)
        rgb_input_image = preprocess_rgb(rgb_image).permute(1, 2, 0).numpy()

        grayscale_cam = gradcam(input_tensor=input_image)
        grayscale_cam = grayscale_cam[0, :]
        output = model(input_image)
        pred = torch.argmax(output, 1)[0]
        visualization = show_cam_on_image(rgb_input_image, grayscale_cam, use_rgb=True)
        visualization = visualize_label(visualization, label, pred)
        plt.imshow(visualization)
        plt.show()
    else:
        image_files = os.listdir(image_dir)
        random.shuffle(image_files)

        visualizations = []
        images = []
        for image_path in image_files[:5]:
            image_name, _ = os.path.splitext(image_path)
            label_file = os.path.join(label_dir, image_name + '.txt')
            if rgb:
                image = Image.open(image_dir + '/' + image_path)
            else:
                image = Image.open(image_dir + '/' + image_path).convert("L")  # Convert to grayscale
            rgb_image = Image.open(image_dir + '/' + image_path)

            #img_aux = cv2.Canny(np.asarray(image), 100, 200)
            #img_aux = cv2.cvtColor(img_aux, cv2.COLOR_GRAY2RGB)
            #input_image = preprocess(img_aux).unsqueeze(0)#.permute(1, 2, 0)#.unsqueeze(0)

            input_image = preprocess(image).unsqueeze(0)
            rgb_input_image = preprocess_rgb(rgb_image).permute(1, 2, 0).numpy()
            output = model(input_image)
            pred = torch.argmax(output, 1)[0]
            with open(label_file, 'r') as file:
                label = file.read()

            attributions = gradcam(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
            attribution = attributions[0, :]
            visualization = show_cam_on_image(rgb_input_image, attribution, use_rgb=True)
            visualization = visualize_label(visualization, str(label), pred)
            images.append(image)
            visualizations.append(visualization)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        plt.subplots_adjust(wspace=0, hspace=0)

        # Plot images
        for i in range(5):
            axes[0, i].imshow(images[i], cmap='gray')  # Assuming images are grayscale
            axes[0, i].axis('off')

        # Plot visualizations
        for i in range(5):
            axes[1, i].imshow(visualizations[i])
            axes[1, i].axis('off')

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or infer a model.')

    # Add arguments for training and inference
    parser.add_argument('--train', action='store_true', help='Train and evaluate the model')
    parser.add_argument('--predict', action='store_true', help='Perform inference using the trained model')
    parser.add_argument('--load', type=str, help='Path to a pre-trained model')
    parser.add_argument('--save', type=str, help='Path where model will be saved')
    parser.add_argument('--image', type=str, help='Path where input image is stored')
    parser.add_argument('--width', type=int, help='Image width')
    parser.add_argument('--height', type=int, help='Image height')
    parser.add_argument('--rgb', action='store_true', help='For RGB training/predictions. Choose an RGB model.')

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
        train_eval_model(df, epochs=10, split=[0.8, 0.2], sample={0: 1000, 1: 1000}, load_path=load_path, save_path=save_path, rgb=args.rgb)
    elif args.predict:
        load_path = None
        if args.load is None:
            parser.error("--load is required when performing inference.")
        if args.width is None:
            parser.error("--width is required when performing inference.")
        if args.height is None:
            parser.error("--height is required when performing inference.")

        predict(args.load, args.width, args.height, args.image, args.rgb)
    else:
        print("Please provide either --train or --predict argument.")
import argparse
import os
import random
from typing import Callable, Union

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
from torchvision import transforms

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, \
    EigenGradCAM, RandomCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50


def visualize_label(visualization, label):
    visualization = cv2.putText(visualization, f"Clase {label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return visualization

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=48, kernel_size=(3, 3), padding="same"),
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

def evaluate_model(model, data_loader):
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

def train_eval_model(df, epochs=None, split=None, sample=None, save_path=None, load_path=None):
    val_split = False
    if split is None:
        split = [0.7, 0.15, 0.15]

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
        X_train = np.array([item for item in df_sample.data.values])
        y_train = np.array(df_sample['label'].values.astype('int'))
    else:
        #X_train = np.array([item.flatten() for item in X_train.data.values])
        X_train = np.array([item for item in X_train.data.values])

    unique, counts = np.unique(y_train, return_counts=True)
    print("Sample:\n", np.asarray((unique, counts)).T)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_train_tensor = X_train_tensor.unsqueeze(1)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Crear conjuntos de datos y DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1] * X_train.shape[2]
    output_size = 2  # Ajusta esto según el número de clases en tu problema

    model = ConvNet(input_size, output_size)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00005)
        if epochs is not None:
            num_epochs = epochs
        else:
            num_epochs = 10
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    # Evaluar el modelo en el conjunto de prueba
    predicted_labels, true_labels = evaluate_model(model, test_loader)

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

    """"
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


def predict(load_path, width, height, image_path=None):
    model = ConvNet(width * height, 2)
    model.load_state_dict(torch.load(load_path))
    target_layers = [model.seq[-4]]
    gradcam = GradCAM(model, target_layers)  # Choose the last convolutional layer
    model.eval()

    image_dir = '../Datasets/Dataset/Femurs/resized_images'
    label_dir = '../Datasets/Dataset/Femurs/augmented_labels_fractura'

    if image_path is not None:
        image = Image.open(image_path).convert("L")
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        label_file = os.path.join(label_dir, image_name + '.txt')
        with open(label_file, 'r') as file:
            label = int(file.read())
        rgb_image = Image.open(image_path)
        transform = transforms.ToTensor()
        input_image = transform(image).unsqueeze(0)
        rgb_input_image = transform(rgb_image)
        rgb_input_image = rgb_input_image.permute(1, 2, 0).numpy()
        grayscale_cam = gradcam(input_tensor=input_image)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_input_image, grayscale_cam, use_rgb=True)
        visualization = visualize_label(visualization, str(label))
        plt.imshow(visualization)
        plt.show()
    else:
        image_files = os.listdir(image_dir)
        random.shuffle(image_files)

        visualizations = []
        for image_path in image_files[:5]:
            image_name, _ = os.path.splitext(image_path)
            label_file = os.path.join(label_dir, image_name + '.txt')
            image = Image.open(image_dir + '/' + image_path).convert("L")  # Convert to grayscale
            rgb_image = Image.open(image_dir + '/' + image_path)
            transform = transforms.ToTensor()
            input_image = transform(image).unsqueeze(0)
            rgb_input_image = transform(rgb_image)
            rgb_input_image = rgb_input_image.permute(1, 2, 0).numpy()

            with open(label_file, 'r') as file:
                label = int(file.read())

            attributions = gradcam(input_tensor=input_image, eigen_smooth=False, aug_smooth=False)
            attribution = attributions[0, :]
            visualization = show_cam_on_image(rgb_input_image, attribution, use_rgb=True)
            visualization = visualize_label(visualization, str(label))
            visualizations.append(visualization)

        plt.imshow(Image.fromarray(np.hstack(visualizations)))
        plt.show()

    """
    # Compute gradients and generate heatmap
    gradcam.compute_gradient(input_image)
    heatmap = gradcam.generate_heatmap().detach().numpy()

    # Resize heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    original_image = cv2.imread(image_path)
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_colormap, 0.4, 0)

    # Display or save the result
    plt.imshow(superimposed_img)
    plt.show()
    """



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

    args = parser.parse_args()

    if args.train:
        load_path = None
        save_path = None
        if args.load:
            load_path = args.load
        if args.save:
            save_path = args.save
        df = pd.read_pickle("../df.pkl")
        train_eval_model(df, split=[0.7, 0.15, 0.15], sample={0: 300, 1: 300}, load_path=load_path, save_path=save_path)
    elif args.predict:
        load_path = None
        if args.load is None:
            parser.error("--load is required when performing inference.")
        if args.width is None:
            parser.error("--width is required when performing inference.")
        if args.height is None:
            parser.error("--height is required when performing inference.")

        predict(args.load, args.width, args.height, args.image)
    else:
        print("Please provide either --train or --predict argument.")
import os

from augment import augment
from create_dataframe import create_dataframe
from model import train_eval_model
from resize import resize
from generate_report import generate_report
import shutil

"""
para tensorboard ir a ./Scripts y ejecutar en la terminal: tensorboard --logdir=./runs y abrir http://localhost:6006/
"""

def combine_datasets(datasets, output_dir):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        input_images_dir = dataset + "/images"
        input_labels_dir = dataset + "/labels"
        output_images_dir = output_dir + "/images"
        output_labels_dir = output_dir + "/labels"

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        for img in os.listdir(input_images_dir):
            shutil.copy(input_images_dir + "/" + img, output_images_dir + "/" + img)
        for label in os.listdir(input_labels_dir):
            shutil.copy(input_labels_dir + "/" + label, output_labels_dir + "/" + label)

if __name__ == "__main__":
    datasets = ["../Datasets/ROB", "../Datasets/AO", "../Datasets/AQ", "../Datasets/MAL"]
    combined_dataset = "../Datasets/COMBINED"

    combine_datasets(datasets, combined_dataset)

    num_classes = 3
    if num_classes == 2:
        sample = {0: 10000, 1: 10000}
        n_augmentations = {0: 1, 1: 7}
    else:
        sample = {0: 10000, 1: 10000, 2: 10000}
        n_augmentations = {0: 0, 1: 3, 2: 3}
    split = [0.8, 0.2]  # 80% train, 20% test
    epochs = 10

    save_report = True
    input_images_folder = combined_dataset + "/images"
    input_labels_folder = combined_dataset + "/labels"
    augmented_images_folder = combined_dataset + "/augmented_images"
    augmented_labels_folder = combined_dataset + "/augmented_labels"
    resized_images_folder = combined_dataset + "/resized_images"
    reports_folder = '../Reports'

    for folder in [augmented_images_folder, augmented_labels_folder, resized_images_folder]:
        shutil.rmtree(folder, ignore_errors=True)


    print("Augmenting images...")
    augment(input_images_folder, input_labels_folder, augmented_images_folder, augmented_labels_folder, n_augmentations,num_classes)
    print("Resizing images...")
    resize(augmented_images_folder, resized_images_folder, padding=False, size=(224, 224))
    print("Creating dataframe...")
    df = create_dataframe(resized_images_folder, augmented_labels_folder, rgb_flag=True, num_classes=num_classes)
    df.to_pickle('../df_rgb.pkl')
    print("Training and evaluating model...")

    report, conf_mat = train_eval_model(df, epochs=epochs, split=split, sample=sample, save_path=f"../models/resnet18_10_3_AO_AQ_MAL", rgb=True, crossval=False, num_classes=num_classes)

    if save_report:
        with open(__file__, 'r') as script_file:
            code = script_file.read()
        generate_report(code, report, conf_mat, reports_folder)
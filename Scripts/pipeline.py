from augment import augment
from create_dataframe import create_dataframe
from model import train_eval_model
from resize import resize
from generate_report import generate_report
import shutil

"""
para tensorboard ejecutar en la terminal: tensorboard --logdir=./runs y abrir http://localhost:6006/
"""

if __name__ == "__main__":
    save_report = True
    input_images_folder = "../Datasets/Dataset/Femurs/grayscale_images"
    input_labels_folder = "../Datasets/Dataset/Femurs/labels_fractura"

    augmented_images_folder = "../Datasets/Dataset/Femurs/augmented_images"
    augmented_labels_folder = "../Datasets/Dataset/Femurs/augmented_labels_fractura"
    resized_images_folder = "../Datasets/Dataset/Femurs/resized_images"
    reports_folder = '../Reports'

    for folder in [augmented_images_folder, augmented_labels_folder, resized_images_folder]:
        shutil.rmtree(folder, ignore_errors=True)

    split = [0.8, 0.2] # 80% train, 20% test

    print("Augmenting images...")
    augment(input_images_folder, input_labels_folder, augmented_images_folder, augmented_labels_folder)
    print("Resizing images...")
    resize(augmented_images_folder, resized_images_folder, padding=False, size=(224, 224))
    print("Creating dataframe...")
    df = create_dataframe(resized_images_folder, augmented_labels_folder, rgb_flag=True)
    df.to_pickle('../df_rgb.pkl')
    print("Training and evaluating model...")
    epochs = 25
    report, conf_mat = train_eval_model(df, epochs=epochs, split=split, sample={0: 100, 1: 100}, save_path=f"../models/resnet18_{epochs}", rgb=True)

    if save_report:
        with open(__file__, 'r') as script_file:
            code = script_file.read()
        generate_report(code, report, conf_mat, reports_folder)
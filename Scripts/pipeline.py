from augment import augment
from create_dataframe import create_dataframe
from model import train_eval_model
from resize import resize
from generate_report import generate_report
import shutil

"""
para tensorboard ir a ./Scripts y ejecutar en la terminal: tensorboard --logdir=./runs y abrir http://localhost:6006/
"""

if __name__ == "__main__":
    num_classes = 2
    if num_classes == 2:
        sample = {0: 10000, 1: 10000}
    else:
        sample = {0: 10000, 1: 10000, 2: 10000}
    split = [0.8, 0.2]  # 80% train, 20% test
    epochs = 10

    save_report = True
    input_images_folder = "../Datasets/Dataset/Femurs/grayscale_images"
    input_labels_folder = "../Datasets/Dataset/Femurs/labels_fractura"

    augmented_images_folder = "../Datasets/Dataset/Femurs/augmented_images"
    augmented_labels_folder = "../Datasets/Dataset/Femurs/augmented_labels_fractura"
    resized_images_folder = "../Datasets/Dataset/Femurs/resized_images"
    reports_folder = '../Reports'

    for folder in [augmented_images_folder, augmented_labels_folder, resized_images_folder]:
        shutil.rmtree(folder, ignore_errors=True)


    print("Augmenting images...")
    augment(input_images_folder, input_labels_folder, augmented_images_folder, augmented_labels_folder, num_classes=num_classes)
    print("Resizing images...")
    resize(augmented_images_folder, resized_images_folder, padding=False, size=(224, 224))
    print("Creating dataframe...")
    df = create_dataframe(resized_images_folder, augmented_labels_folder, rgb_flag=True, num_classes=num_classes)
    df.to_pickle('../df_rgb.pkl')
    print("Training and evaluating model...")

    report, conf_mat = train_eval_model(df, epochs=epochs, split=split, sample=sample, save_path=f"../models/resnet18_10_2_AO_AQ_MAL", rgb=True, crossval=False, num_classes=num_classes)

    if save_report:
        with open(__file__, 'r') as script_file:
            code = script_file.read()
        generate_report(code, report, conf_mat, reports_folder)
from augment import augment
from resize import resize
from create_dataframe import create_dataframe
from model import train_eval_model

if __name__ == "__main__":
    input_images_folder = "./Datasets/Dataset/Femurs/grayscale_images"
    input_labels_folder = "./Datasets/Dataset/Femurs/labels_fractura"

    augmented_images_folder = "./Datasets/Dataset/Femurs/augmented_images"
    augmented_labels_folder = "./Datasets/Dataset/Femurs/augmented_labels_fractura"

    resized_images_folder = "./Datasets/Dataset/Femurs/resized_images"

    split = [0.7, 0.15, 0.15]  # train, test, val

    print("Augmenting images...")
    augment(input_images_folder, input_labels_folder, augmented_images_folder, augmented_labels_folder)
    print("Resizing images...")
    resize(augmented_images_folder, resized_images_folder, padding=True)
    print("Creating dataframe...")
    df = create_dataframe(resized_images_folder, augmented_labels_folder)
    df.to_pickle('df.pkl')
    print("Training and evaluating model...")
    train_eval_model(df, split)

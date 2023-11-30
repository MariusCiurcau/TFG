import os
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import TextArea, AnnotationBbox
from PIL import Image


def plot_image_with_label(row):
    # Read image
    image_path = row['filename']
    image = Image.open(image_path)

    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    # Add label below the image
    label = row['label']
    label_box = TextArea(f'Label: {label}', textprops=dict(color='black', size=10))
    ab = AnnotationBbox(label_box, (0.5, -0.05), frameon=False, boxcoords="axes fraction")
    ax.add_artist(ab)

    # Show the plot
    plt.show()

def df_from_folders(images_folder, labels_folder):
    data = dict()
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    for img in os.listdir(images_folder):
        im = imread(os.path.join(images_folder, img), as_gray=True)
        label_path = os.path.join(labels_folder, os.path.splitext(img)[0] + '.txt')
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label = int((file.readlines()[0]).split()[0])
        else:
            print(f"Image: {img}, label file not found.")

        data['label'].append(label)
        data['filename'].append(os.path.join(images_folder, img))
        data['data'].append(im)

    df = pd.DataFrame.from_dict(data)
    return df

def create_dataframes(train_folder, test_folder, val_folder, use_augmented):
    augmented_prefix = ''
    if use_augmented:
        augmented_prefix = 'augmented_'
    train_images_folder = os.path.join(train_folder, augmented_prefix + 'images')
    train_labels_folder = os.path.join(train_folder, augmented_prefix + 'labels')
    df_train = df_from_folders(train_images_folder, train_labels_folder)

    test_images_folder = os.path.join(test_folder, 'images')
    test_labels_folder = os.path.join(test_folder, 'labels')
    df_test = df_from_folders(test_images_folder, test_labels_folder)

    val_images_folder = os.path.join(val_folder, 'images')
    val_labels_folder = os.path.join(val_folder, 'labels')
    df_val = df_from_folders(val_images_folder, val_labels_folder)

    return df_train, df_test, df_val

def flatten_array(array):
    return array.flatten()

def main():
    use_augmented = True
    train_folder = "./Datasets/Dataset/Femurs/padded_split/train"
    test_folder = "./Datasets/Dataset/Femurs/padded_split/test"
    val_folder = "./Datasets/Dataset/Femurs/padded_split/val"
    df_train, df_test, df_val = create_dataframes(train_folder, test_folder, val_folder, use_augmented)
    df_train.to_pickle('df_train.pkl')
    df_train.to_pickle('df_test.pkl')
    df_train.to_pickle('df_val.pkl')

if __name__ == "__main__":
    main()
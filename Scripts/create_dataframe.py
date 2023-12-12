import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.offsetbox import TextArea, AnnotationBbox
from skimage.io import imread


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


def flatten_array(array):
    return array.flatten()


def create_dataframe(images_folder, labels_folder):
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


if __name__ == "__main__":
    images_folder = "./Datasets/Dataset/Femurs/resized_augmented_images"
    labels_folder = "./Datasets/Dataset/Femurs/augmented_labels_fractura"
    df = create_dataframe(images_folder, labels_folder)
    df.to_pickle('df.pkl')

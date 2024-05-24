import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.offsetbox import TextArea, AnnotationBbox
from skimage.io import imread

from utils import read_label


def plot_image_with_label(row):
    image_path = row['filename']
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    label = row['label']
    label_box = TextArea(f'Label: {label}', textprops=dict(color='black', size=10))
    ab = AnnotationBbox(label_box, (0.5, -0.05), frameon=False, boxcoords="axes fraction")
    ax.add_artist(ab)
    plt.show()


def flatten_array(array):
    return array.flatten()


def create_dataframe(images_folder, labels_folder, rgb_flag, num_classes=2):
    data = dict()
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    for img in os.listdir(images_folder):
        im = imread(os.path.join(images_folder, img), as_gray=not rgb_flag)
        label_path = os.path.join(labels_folder, os.path.splitext(img)[0] + '.txt')

        if os.path.exists(label_path):
            label = read_label(label_path, num_classes)
            data['label'].append(label)
            data['filename'].append(os.path.join(images_folder, img))
            data['data'].append(im)
        else:
            print(f"Image: {img}, label file not found.")

    df = pd.DataFrame.from_dict(data)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose to create a grayscale or rgb dataframe.')

    parser.add_argument('--rgb', action='store_true', help='For RGB DataFrame.')

    args = parser.parse_args()

    images_folder = "./Datasets/Dataset/Femurs/resized_images"
    labels_folder = "./Datasets/Dataset/Femurs/augmented_labels_fractura"
    
    df = create_dataframe(images_folder, labels_folder, rgb_flag=args.rgb)
    if args.rgb:
        df.to_pickle('df_rgb.pkl')
    else:
        df.to_pickle('df.pkl')
    print(df.info())
    print(df['data'].shape)

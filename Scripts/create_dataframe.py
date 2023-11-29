import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

def create_dataframe(data_folder, width=200, height=None):
    height = height if height is not None else width
    data = dict()
    data['description'] = 'resized ({0}x{1}) fracture images in rgb'.format(int(width), int(height))
    data['label'] = []
    #data['filename'] = []
    data['data'] = []

    #pklname = f"{pklname}_{width}x{height}px.pkl"

    images_folder = os.path.join(data_folder, 'images')
    labels_folder = os.path.join(data_folder, 'labels_fractura')
    print(labels_folder)

    for file in os.listdir(images_folder):
        im = imread(os.path.join(images_folder, file))
        im = resize(im, (width, height))  # [:,:,::-1]
        label_path = os.path.join(labels_folder, os.path.splitext(file)[0] + '.txt')
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label = int((file.readlines()[0]).split()[0])
        else:
            print(f"Image: {file}, label file not found.")

        data['label'].append(label)
        #data['filename'].append(file)
        data['data'].append(im)

    df = pd.DataFrame.from_dict(data)
    return df

    #joblib.dump(data, pklname)
    # Store data (serialize)
    #with open(os.path.join(destination_folder, f'{pklname}.pickle'), 'wb') as file:
    #    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    data_folder = "./Datasets/Dataset/Femurs"


    df = create_dataframe(data_folder)
    pklname = 'df.pkl'
    df.to_pickle(pklname)

    print(df.head())

    """"
    data = joblib.load(f'{pklname}.pkl')
    labels = np.unique(data['label'])

    # set up the matplotlib figure and axes, based on the number of labels
    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15, 4)
    fig.tight_layout()

    # make a plot for every label (equipment) type. The index method returns the
    # index of the first item corresponding to its search string, label in this case
    for ax, label in zip(axes, labels):
        idx = data['label'].index(label)

        ax.imshow(data['data'][idx])
        ax.axis('off')
        ax.set_title(label)
    """
if __name__ == "__main__":
    main()
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

def create_dataframe(data_folder, width=200, height=None):
    height = height if height is not None else width
    data = dict()
    data['description'] = 'resized ({0}x{1}) grayscale fracture images'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    #pklname = f"{pklname}_{width}x{height}px.pkl"

    images_folder = os.path.join(data_folder, 'padded_augmented_images')
    labels_folder = os.path.join(data_folder, 'augmented_labels_fractura')

    for img in os.listdir(images_folder):
        im = imread(os.path.join(images_folder, img), as_gray=True)
        #im = resize(im, (width, height))  # [:,:,::-1]
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

    #joblib.dump(data, pklname)
    # Store data (serialize)
    #with open(os.path.join(destination_folder, f'{pklname}.pickle'), 'wb') as file:
    #    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def flatten_array(array):
    return array.flatten()

def main():
    data_folder = "./Datasets/Dataset/Femurs"
    df = create_dataframe(data_folder, 183, 299)
    pklname = 'df.pkl'
    df.to_pickle(pklname)

if __name__ == "__main__":
    main()
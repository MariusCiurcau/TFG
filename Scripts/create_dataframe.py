import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import TextArea, OffsetImage, AnnotationBbox
from PIL import Image
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

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

    images_folder = os.path.join(data_folder, 'padded_images')
    labels_folder = os.path.join(data_folder, 'labels_fractura')

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
    print(df['data'][0].shape)
    # Apply the function to the DataFrame column
    df['data'] = df['data'].apply(flatten_array)
    pklname = 'df.pkl'
    #df.to_pickle(pklname)

    print(df.head())
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    

    # Iterate through the first 5 rows of the DataFrame
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        # Read image
        image_path = row['filename']
        image = Image.open(image_path)

        # Plot image
        axes[i].imshow(image)
        axes[i].axis('off')

        # Add label below the image
        label = row['label']
        axes[i].text(0.5, -0.05, f'Label: {label}', color='black', size=10, ha='center', va='center', transform=axes[i].transAxes)

    # Adjust layout and show the plot
    plt.tight_layout()
    #plt.show()

    clf = svm.SVC(gamma=0.001)

    digits = datasets.load_digits()
    print(digits.images[0])

    n_samples = len(df.label)
    flattened_data = np.array([item.flatten() for sublist in df.data.values for item in sublist])


    print(flattened_data.shape)
    print(df.label.values.shape)
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        flattened_data, df.label.values, test_size=0.2, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

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
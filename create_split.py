import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

def series_to_folder(series, input_images_folder, input_labels_folder, output_images_folder, output_labels_folder):
    for filename in series:
        input_image_path = os.path.join(input_images_folder, filename)
        input_label_path = os.path.join(input_labels_folder, os.path.splitext(filename)[0] + '.txt')
        output_image_path = os.path.join(output_images_folder, filename)
        output_label_path = os.path.join(output_labels_folder, os.path.splitext(filename)[0] + '.txt')
        shutil.copy(input_image_path, output_image_path)
        shutil.copy(input_label_path, output_label_path)

def create_split(input_images_folder, input_labels_folder, output_folder):
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    val_folder = os.path.join(output_folder, 'val')
    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'labels'), exist_ok=True)

    data = dict()
    data['label'] = []
    data['filename'] = []
    #data['data'] = []

    for img in os.listdir(input_images_folder):
        label_path = os.path.join(input_labels_folder, os.path.splitext(img)[0] + '.txt')
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label = int((file.readlines()[0]).split()[0])
        else:
            print(f"Image: {img}, label file not found.")

        data['label'].append(label)
        data['filename'].append(img)
        #data['data'].append(im)

    df = pd.DataFrame.from_dict(data)

    X_aux, X_val, y_aux, y_val = train_test_split(df.filename, df.label, test_size=0.15, shuffle=True, random_state=1, stratify=df.label)

    X_train, X_test, y_train, y_test = train_test_split(X_aux, y_aux, test_size=0.15/0.85, shuffle=True, random_state=1, stratify=y_aux)

    series_to_folder(X_train, input_images_folder, input_labels_folder, os.path.join(train_folder, 'images'), os.path.join(train_folder, 'labels'))
    series_to_folder(X_test, input_images_folder, input_labels_folder, os.path.join(test_folder, 'images'), os.path.join(test_folder, 'labels'))
    series_to_folder(X_val, input_images_folder, input_labels_folder, os.path.join(val_folder, 'images'), os.path.join(val_folder, 'labels'))

def main():
    input_images_folder = "./Datasets/Dataset/Femurs/grayscale_images"
    input_labels_folder = "./Datasets/Dataset/Femurs/labels_fractura"
    output_folder = "./Datasets/Dataset/Femurs/split"
    create_split(input_images_folder, input_labels_folder, output_folder)

if __name__ == "__main__":
    main()
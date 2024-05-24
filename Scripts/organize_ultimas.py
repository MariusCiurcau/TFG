import os
import shutil

import cv2


def organize_images_and_labels(i, image_dir, label, output_dir, source_name):
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Iterate through image files
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                image_name, _ = os.path.splitext(os.path.basename(file))

                image_path = os.path.join(root, file)

                original_image = cv2.imread(image_path)
                grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                equalized_image = cv2.equalizeHist(grayscale_image)
                equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

                # resize image (for Roboflow images)
                #pil = Image.open(image_path).convert('RGB')
                #image = np.array(pil)
                image = cv2.resize(equalized_image, (224, 224))

                image_destination_path = os.path.join(output_image_dir, f"{source_name}_{i:04d}.jpg")
                label_destination_path = os.path.join(output_label_dir, f"{source_name}_{i:04d}.txt")

                cv2.imwrite(image_destination_path, image)
                with open(label_destination_path, 'w') as f:
                    f.write(str(label))
            i += 1
    return i

def main():
    image_dirs = ['../Datasets/FXMalaga/ULTIMAS/Sin fractura', '../Datasets/FXMalaga/ULTIMAS/Fx cuello', '../Datasets/FXMalaga/ULTIMAS/Fx pertro']
    labels = [0, 1, 2]

    output_dir = f'../Datasets/ULTIMAS'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    i = 0
    for image_dir, label in zip(image_dirs, labels):
        i = organize_images_and_labels(i, image_dir, label, output_dir, source_name='ULTIMAS')

if __name__ == '__main__':
    main()
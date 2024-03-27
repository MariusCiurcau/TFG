import os
import shutil
import unicodedata

def normalize_string(string):
    # Replace accents with corresponding base characters
    normalized_string = ''.join(c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn')
    # Replace spaces with underscores
    normalized_string = normalized_string.replace(" ", "_")
    return normalized_string
def organize_images_and_labels(image_dir, label_dir, output_dir):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Iterate through image files
    i = 0
    last_dir = os.path.basename(os.path.normpath(output_dir))
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                image_name, _ = os.path.splitext(os.path.basename(file))
                image_name_normalized = normalize_string(image_name)

                image_path = os.path.join(root, file)
                print(image_path)
                label_path = os.path.join(label_dir, image_name_normalized + '.txt')

                #image_file_normalized = normalize_string(file)
                #image_destination_path = os.path.join(output_image_dir, image_file_normalized)
                image_destination_path = os.path.join(output_image_dir, f"{last_dir}_{i:04d}.jpg")
                label_destination_path = os.path.join(output_label_dir, f"{last_dir}_{i:04d}.txt")

                if os.path.exists(label_path):
                    shutil.copy(image_path, image_destination_path)
                    shutil.copy(label_path, label_destination_path)
                else:
                    print(label_path)
                    print(f"Label file for {file} not found.")
            i += 1

if __name__ == "__main__":
    images_dirs = ["../Datasets/Dataset/Femurs/Versiones anteriores/grayscale_images_old", "../Datasets/original_AO/resized_images",
                   "../Datasets/FXMalaga/resized_images", "../Datasets/FracturasAQ/Data/resized_images"]
    labels_dir = "../Datasets/Dataset/Femurs/labels/3clases/labels_fractura"
    output_dirs = ["../Datasets/ROB", "../Datasets/AO", "../Datasets/MAL", "../Datasets/AQ"]

    for images_dir, output_dir in zip(images_dirs, output_dirs):
        organize_images_and_labels(images_dir, labels_dir, output_dir)
import os


def rename(image_dir, label_dir, prefix):
    # Iterate through image files
    for root, _, files in os.walk(image_dir):
        i = 0
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                image_name, _ = os.path.splitext(os.path.basename(file))

                image_path = os.path.join(root, file)
                label_path = os.path.join(label_dir, image_name + '.txt')

                os.rename(image_path, os.path.join(image_dir, f"{prefix}_{i:04d}.jpg"))
                os.rename(label_path, os.path.join(label_dir, f"{prefix}_{i:04d}.txt"))
                i += 1

def main():
    image_dir = '../Datasets/ULTIMAS/images'
    label_dir = '../Datasets/ULTIMAS/labels'
    rename(image_dir, label_dir, 'ULTIMAS')

if __name__ == '__main__':
    main()
import os

from PIL import Image


def convert_tif_to_jpg_recursive(input_folder):
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.tif') or filename.endswith('.tiff') or filename.endswith('.TIF'):
                input_path = os.path.join(root, filename)
                try:
                    img = Image.open(input_path)
                    output_path = os.path.splitext(input_path)[0] + '.jpg'
                    img.convert('RGB').save(output_path, 'JPEG')
                    print(f"Converted {filename} and overwrote the original file")
                except Exception as e:
                    print(f"Error converting {filename}: {e}")


# Replace '/path/to/input/folder' with your actual path
folder_path = '../Datasets/AO/31-/'

convert_tif_to_jpg_recursive(folder_path)

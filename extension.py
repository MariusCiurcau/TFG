import os

def add_extension_to_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            name, extension = os.path.splitext(filename)

            if extension != '.jpg':
                new_name = name + extension + '.jpg'
                new_path = os.path.join(root, new_name)
                os.rename(file_path, new_path)
                print(f"Added .jpg extension to {filename}")

def fix_jpg(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            name, extension = os.path.splitext(filename)

            if name.endswith('.TIF') or name.endswith('.tif'):
                new_name = name[:-4] + extension
                new_path = os.path.join(root, new_name)
                os.rename(file_path, new_path)
                print(f"Renamed {filename}")
# Replace '/path/to/your/folder' with the actual path to your folder
folder_path = './Datasets/AO/31-/'
#add_extension_to_files(folder_path)

fix_jpg(folder_path)
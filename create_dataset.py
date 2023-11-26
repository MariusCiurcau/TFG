import os
import subprocess

def run_yolov5_detection(image_path):
    # Assuming yolov5 is installed in the same directory as this script
    yolov5_script_path = './yolov5_ws/yolov5/detect.py'

    # Run YOLOv5 detection
    subprocess.run(['python', yolov5_script_path, '--source', image_path, '--save-crop', '--project', 'Datasets/Clasificador', '--exist-ok'])

def read_label_from_file(label_path):
    with open(label_path, 'r') as file:
        label = file.read().strip()
        return label

def main():
    image_folder = './Datasets/Proximal Femur Fracture.v11i.yolov5pytorch/train/images'
    label_folder = './Datasets/Proximal Femur Fracture.v11i.yolov5pytorch/train/labels'

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add more image extensions if needed
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')

            # Run YOLOv5 detection
            run_yolov5_detection(image_path)

            # Read label from the corresponding text file
            if os.path.exists(label_path):
                label = read_label_from_file(label_path)
                print(f"Image: {filename}, Label: {label}")
            else:
                print(f"Image: {filename}, Label file not found.")

if __name__ == "__main__":
    main()

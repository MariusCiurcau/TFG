import os
import subprocess

def run_yolov5_detection(folder_path, weights_path):
    # Assuming yolov5 is installed in the same directory as this script
    yolov5_script_path = './yolov5_ws/yolov5/detect.py'

    # Run YOLOv5 detection
    subprocess.run(['python', yolov5_script_path, '--source', folder_path, '--save-crop', '--project', 'Datasets/Dataset/', '--name', 'Femurs', '--exist-ok', '--weights', weights_path, '--classes', '0', '--max-det', '2'])

fracture_map = {0: 'dislocation', 1: 'grater-trochanter', 2: 'intertrochanteric', 3: 'lesser-trochanter', 4: 'neck', 5: 'normal', 6: 'subtrochanteric'}

def create_new_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
        max = 0
        min = 1
        tipo_max = 5
        tipo_min = 5
        for line in lines:
            values = line.split()
            tipo = int(values[0])
            coord = float(values[1])
            if coord < min:
                min = coord
                tipo_min = tipo
            if coord > max:
                max = coord
                tipo_max = tipo
        label1 = label2 = "1"
        if tipo_min == 5:
            label1 = "0"
        elif tipo_min == 0: # flag dislocation
            label1 = "-1"
        if tipo_max == 5:
            label2 = "0"
        elif tipo_max == 0: # flag dislocation
            label2 = "-1"
        return label1, label2

def main():
    image_folder = './Datasets/Dataset/Proximal/Bilateral/images'
    label_folder = './Datasets/Dataset/Proximal/Bilateral/labels'
    femurs_folder = './Datasets/Dataset/Femurs'
    femurs_images_folder = femurs_folder / 'images'
    femurs_labels_folder = femurs_folder / 'labels'
    weights_path = './yolov5_ws/yolov5/runs/train/exp3/weights/best.pt'

    run_yolov5_detection(folder_path, weights_path)

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add more image extensions if needed
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')

            # Read label from the corresponding text file
            if os.path.exists(label_path):
                label1, label2 = create_new_labels(label_path)
                print(f"Image: {filename}, Labels: {label1, label2}")
            else:
                print(f"Image: {filename}, Label file not found.")
            
            if (os.path.exists(os.path.join(femurs_images_folder, os.path.splitext(filename)[0] + '.jpg'))):
                file1 = open(os.path.join(femurs_labels_folder, os.path.splitext(filename)[0] + '.txt'), 'w+')
                file1.write(label1)
                file1.close()
                if (os.path.exists(os.path.join(femurs_images_folder, os.path.splitext(filename)[0] + '2.jpg'))):
                    file2 = open(os.path.join(femurs_labels_folder, os.path.splitext(filename)[0] + '2.txt'), 'w+')
                    file2.write(label2)
                    file2.close()

if __name__ == "__main__":
    main()

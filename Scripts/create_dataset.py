import os
import subprocess


def run_yolov5_detection(folder_path, weights_path):
    # Assuming yolov5 is installed in the same directory as this script
    yolov5_script_path = '../yolov5_ws/yolov5/detect.py'

    # Run YOLOv5 detection
    subprocess.run(
        ['python', yolov5_script_path, '--source', folder_path, '--save-crop', '--project', 'Datasets/Dataset/',
         '--name', 'Femurs', '--exist-ok', '--weights', weights_path, '--classes', '0', '--max-det', '2', '--save-txt',
         '--save-conf'])


fracture_map = {0: 'dislocation', 1: 'grater-trochanter', 2: 'intertrochanteric', 3: 'lesser-trochanter', 4: 'neck',
                5: 'normal', 6: 'subtrochanteric'}


def create_new_labels(label_path, detection_label_path):
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
        label1 = "1"
        label2 = "1"
        if tipo_min == 5:
            label1 = "0"
        elif tipo_min == 0:  # flag dislocation
            label1 = "-1"
        if tipo_max == 5:
            label2 = "0"
        elif tipo_max == 0:  # flag dislocation
            label2 = "-1"
    with open(detection_label_path, 'r') as file:
        lines = file.readlines()
        max_conf = 0
        max = 0
        min = 1
        coord_max_conf = 0
        for line in lines:
            values = line.split()
            coord = float(values[1])
            conf = float(values[-1])
            if conf > max_conf:
                max_conf = conf
                coord_max_conf = coord
            if coord < min:
                min = coord
            if coord > max:
                max = coord
        if coord_max_conf == min:
            label1, label2 = label2, label1
            print(f"Inverted labels for {detection_label_path}")
    return label1, label2


def run_labelling(femurs_images_folder, femurs_labels_folder, detection_labels_folder, image_folder, label_folder):
    if not os.path.exists(femurs_labels_folder):
        os.makedirs(femurs_labels_folder)
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
            detection_label_path = os.path.join(detection_labels_folder, os.path.splitext(filename)[0] + '.txt')

            # Read label from the corresponding text file
            if os.path.exists(label_path) and os.path.exists(detection_label_path):
                label1, label2 = create_new_labels(label_path, detection_label_path)
                print(f"Image: {filename}, Labels: {label1, label2}")
                if os.path.exists(os.path.join(femurs_images_folder, os.path.splitext(filename)[0] + '.jpg')):
                    if label1 == "-1":
                        os.remove(os.path.join(femurs_images_folder, os.path.splitext(filename)[0] + '.jpg'))
                    else:
                        file1 = open(os.path.join(femurs_labels_folder, os.path.splitext(filename)[0] + '.txt'), 'w+')
                        file1.write(label1)
                        file1.close()
                if os.path.exists(os.path.join(femurs_images_folder, os.path.splitext(filename)[0] + '2.jpg')):
                    if label2 == "-1":
                        os.remove(os.path.join(femurs_images_folder, os.path.splitext(filename)[0] + '2.jpg'))
                    else:
                        file2 = open(os.path.join(femurs_labels_folder, os.path.splitext(filename)[0] + '2.txt'), 'w+')
                        file2.write(label2)
                        file2.close()
            else:
                print(f"Image: {filename}, Label file not found.")


def move_detections(femurs_folder):
    detections_folder = os.path.join(femurs_folder, 'detections')
    if not os.path.exists(detections_folder):
        os.makedirs(detections_folder)
    for filename in os.listdir(femurs_folder):
        if filename.endswith(('.jpg')):
            os.rename(os.path.join(femurs_folder, filename), os.path.join(detections_folder, filename))


def main():
    images_folder = './Datasets/Dataset/Proximal/Bilateral/images'
    labels_folder = './Datasets/Dataset/Proximal/Bilateral/labels'
    femurs_folder = './Datasets/Dataset/Femurs'
    detection_labels_folder = os.path.join(femurs_folder, 'labels')
    femurs_images_folder = os.path.join(femurs_folder, 'images')
    femurs_labels_folder = os.path.join(femurs_folder, 'labels_fractura')
    weights_path = '../yolov5_ws/yolov5/runs/train/exp3/weights/best.pt'

    if not os.path.exists(femurs_images_folder):
        run_yolov5_detection(images_folder, weights_path)
        move_detections(femurs_folder)
    else:
        print("images folder already exists. Skipping detection...")

    run_labelling(femurs_images_folder, femurs_labels_folder, detection_labels_folder, images_folder, labels_folder)


if __name__ == "__main__":
    main()

import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity

def visualize_label(visualization, label, prediction, score=None, name=None, similar=False, filename=None):
    filename_offset = 0
    if filename is not None:
        filename_offset = 30
        visualization = cv2.putText(visualization, f"{filename}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    label_offset = 0
    if label is not None:
        visualization = cv2.putText(visualization, f"Class: {label}", (10, 30 + filename_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        label_offset = 30
    visualization = cv2.putText(visualization, f"Prediction: {prediction}", (10, 30 + filename_offset + label_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if score is not None:
        visualization = cv2.putText(visualization, f"Score: {score:.3f}", (10, 60 + filename_offset + label_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if similar:
        visualization = cv2.putText(visualization, f"Similar", (10, 60 + filename_offset + label_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if name is not None:
        visualization = cv2.putText(visualization, f"Method: {name}", (10, 90 + filename_offset + label_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return visualization

def add_filename(image, image_name):
    image = cv2.putText(image, f"{image_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def add_border(visualization, label, pred):
    false_positive = (label == 0) and (pred != 0)
    false_negative = (label != 0) and (pred == 0)
    correct = label == pred
    if false_positive:
        visualization = cv2.copyMakeBorder(visualization, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 255))
    elif false_negative:
        visualization = cv2.copyMakeBorder(visualization, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    elif correct:
        visualization = cv2.copyMakeBorder(visualization, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    return visualization

def find_similar_images(image_path, image_label, image_files, images_dir, labels_dir, num_images, num_classes):
    image_files = [image_file for image_file in image_files if
                   image_file != image_path and image_file.endswith('_0.jpg')]
    image_files_same_label = []
    image_files_different_label = []

    for image_file in image_files:
        label_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')
        label = read_label(label_file, num_classes=num_classes)
        if label == image_label:
            image_files_same_label.append(image_file)
        else:
            image_files_different_label.append(image_file)

    img = cv2.imread(images_dir + '/' + image_path, cv2.IMREAD_GRAYSCALE)
    range_img = img.max() - img.min()
    ssims = []

    for image_file in image_files_same_label:
        img_aux = cv2.imread(images_dir + '/' + image_file, cv2.IMREAD_GRAYSCALE)
        if image_file != image_path:
            range_ = max(range_img, img_aux.max() - img_aux.min())
            ssim = structural_similarity(img, img_aux, data_range=range_)
            ssims.append(ssim)
            """
            if ssim > best_ssim:
                best_ssim = ssim
                best_image_file = image_file
            """
    best_ssims = np.argpartition(ssims, -num_images)[-num_images:]
    # print(best_ssims)
    best_image_files = np.array(image_files_same_label)[best_ssims]
    # print(best_image_files)
    print("Similar images to " + image_path + " found:", best_image_files)
    #print("SSIMs:", best_ssims)
    return best_image_files

def read_label(label_file, num_classes=2):
    with open(label_file, 'r') as file:
        label = int(file.read())
        if label > num_classes - 1:
            label = num_classes - 1
    return label

def show_cam_on_image_alpha(img, mask, **kwargs):
    mask = mask ** 2
    mask = mask / mask.max()
    #cam_alpha = np.uint8(255 * mask)  # Convert mask to uint8
    cam_alpha = cv2.merge([mask, mask, mask])  # Create 3-channel alpha mask
    #cam_alpha = np.float32(cam_alpha) / 255.0  # Normalize to [0, 1]

    blended_img = img * cam_alpha

    return np.uint8(255 * blended_img)
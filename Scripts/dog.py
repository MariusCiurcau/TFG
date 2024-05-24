import cv2
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('../Datasets/Dataset/Femurs/resized_images/24_png.rf.997bf77880b5fdcad3c21915d8aa3ade_1.jpg', cv2.IMREAD_COLOR)
img = img / 255.0

# Define the number of radii
num_radii = 15

# Create a figure with subplots
fig, axes = plt.subplots(num_radii, num_radii, figsize=(20, 20), dpi=600)

# Iterate over all combinations of radius1 and radius2
for i, radius1 in enumerate(range(1, num_radii + 1)):
    for j, radius2 in enumerate(range(1, num_radii + 1)):
        # Skip combinations where radius1 >= radius2
        if radius1 >= radius2:
            continue

        # Compute sigmas based on radii
        sigma1 = 0.3003866304138461 * (radius1 + 1.0)
        sigma2 = 0.3003866304138461 * (radius2 + 1.0)

        # Apply Gaussian blur with the calculated sigmas
        gaussian_blur1 = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
        gaussian_blur2 = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma2, sigmaY=sigma2)

        # Calculate the Difference of Gaussians
        dog = gaussian_blur2 - gaussian_blur1

        # Plot the DoG image
        ax = axes[i, j]
        ax.imshow(dog * 255)
        ax.axis('off')
        ax.set_title(f"R1: {radius1}, R2: {radius2}")

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('dog.png')
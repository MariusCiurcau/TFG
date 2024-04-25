import os
import random

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 18
rc_params = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "text.usetex": True
}
matplotlib.rcParams.update(rc_params)
plt.rc('text.latex', preamble=r'\usepackage{libertine}\usepackage[T1]{fontenc}')

IMAGES_PATH = '../Datasets/COMBINED/resized_images'
LABELS_PATH = '../Datasets/COMBINED/augmented_labels'
CLUSTERS_PATH = "../Datasets/Dataset/Femurs/clusters"


dirs = ["../Datasets/Dataset/Femurs/clusters/label0",
        "../Datasets/Dataset/Femurs/clusters/label1",
        "../Datasets/Dataset/Femurs/clusters/label2"]
num_clusters = [1, 3, 2]
cluster_paths = ["../Datasets/Dataset/Femurs/clusters/label0/cluster0",
                 "../Datasets/Dataset/Femurs/clusters/label1/cluster0",
                 "../Datasets/Dataset/Femurs/clusters/label1/cluster1",
                 "../Datasets/Dataset/Femurs/clusters/label1/cluster2",
                 "../Datasets/Dataset/Femurs/clusters/label2/cluster0",
                 "../Datasets/Dataset/Femurs/clusters/label2/cluster1"]
inits = [["../Datasets/Dataset/Femurs/clusters/label0/c0.jpg"],
         ["../Datasets/Dataset/Femurs/clusters/label1/c0.jpg", "../Datasets/Dataset/Femurs/clusters/label1/c1.jpg", "../Datasets/Dataset/Femurs/clusters/label1/c2.jpg"],
         ["../Datasets/Dataset/Femurs/clusters/label2/c0.jpg", "../Datasets/Dataset/Femurs/clusters/label2/c1.jpg"]]

len_clusters = {}
cluster_images = {}
for dir in cluster_paths:
    parts = dir.split('/')
    cluster = int(parts[-1][-1])
    label = int(parts[-2][-1])
    images = [x for x in os.listdir(dir) if x.endswith(('_0.jpg', '_0.jpeg', '_0.png'))]
    len_clusters[f'C{label}.{cluster}'] = len(images)
    cluster_images[f'C{label}.{cluster}'] = '../Datasets/Dataset/Femurs/clusters/label' + str(label) + f'/c{cluster}.jpg'

print(len_clusters)
print(cluster_images)
np.random.seed(43)
locations = {
    'C0.0': (0, 0),
    'C1.0': (0.8, 1),
    'C1.1': (0.2, 1),
    'C1.2': (0.5, 0.7),
    'C2.0': (0.9, 0),
    'C2.1': (0.7, -0.4)
}

colors = {
    'C0.0': 'darkgreen',
    'C1.0': '#8080c5',
    'C1.1': '#4d4dae',
    'C1.2': 'darkblue',
    'C2.0': '#ae4d4d',
    'C2.1': 'darkred'
}
fig, ax = plt.subplots(figsize=(12, 12))
max_num_images = max(len_clusters.values())
min_x, max_x = 0, 1
min_y, max_y = 0, 1
for cluster, num_images in len_clusters.items():
    locx, locy = locations[cluster]
    if num_images < 0.2*max_num_images:
        num_images = int(0.2*max_num_images)
    x = np.random.normal(loc=locx, scale=0.1, size=num_images)
    y = np.random.normal(loc=locy, scale=0.1, size=num_images)
    min_x = min(min_x, np.min(x))
    max_x = max(max_x, np.max(x))
    min_y = min(min_y, np.min(y))
    max_y = max(max_y, np.max(y))
    clase = int(cluster[1])
    clust = int(cluster[-1])
    plt.scatter(x, y, color=colors[cluster], s=15, label=f'Class {clase}, cluster {clust} (C{clase}.{clust})', zorder=1)
    x_center = np.mean(x)
    y_center = np.mean(y)
    img = plt.imread(cluster_images[cluster])
    plt.imshow(img, extent=(x_center - 0.0007*img.shape[1]/2, x_center + 0.0007*img.shape[1]/2, y_center-0.0007*img.shape[0]/2, y_center + 0.0007*img.shape[0]/2), zorder=2)
plt.xlim(min_x - 0.01, max_x + 0.3)
plt.ylim(min_y - 0.01, max_y + 0.01)
ax.set_axis_off()
lgnd = plt.legend(loc='upper right', bbox_to_anchor=(1, 0.75), frameon=False, markerscale=2)
plt.tight_layout()
plt.savefig('../figures/clusters.pdf', bbox_inches='tight')
plt.show()

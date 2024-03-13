


# Copyright (c) 2016, Oleg Puzanov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import datetime
import pickle
import cv2
import numpy as np
import ssim.ssim as pyssim
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import SpectralClustering, AffinityPropagation, DBSCAN, KMeans
from sklearn import metrics
import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import shutil
import matplotlib.pyplot as plt


# Constant definitions
SIM_IMAGE_SIZE = (224, 224)
SIFT_RATIO = 0.7
MSE_NUMERATOR = 1000.0
IMAGES_PER_CLUSTER = 2


dirs = ["../Datasets/Dataset/Femurs/textos/label0", "../Datasets/Dataset/Femurs/textos/label1", "../Datasets/Dataset/Femurs/textos/label2"]
num_clusters = [1,3,2]
inits  = [["../Datasets/Dataset/Femurs/textos/label0/c0.jpg"],
            ["../Datasets/Dataset/Femurs/textos/label1/c0.jpg", "../Datasets/Dataset/Femurs/textos/label1/c1.jpg", "../Datasets/Dataset/Femurs/textos/label1/c2.jpg"], 
            ["../Datasets/Dataset/Femurs/textos/label2/c0.jpg","../Datasets/Dataset/Femurs/textos/label2/c1.jpg"]]




""" Returns the normalized similarity value (from 0.0 to 1.0) for the provided pair of images.
    The following algorithms are supported:
    * SIFT: Scale-invariant Feature Transform
    * SSIM: Structural Similarity Index
    * CW-SSIM: Complex Wavelet Structural Similarity Index
    * MSE: Mean Squared Error
"""
def get_image_similarity(img1, img2, algorithm='SSIM'):
    # Converting to grayscale and resizing
    i1 = cv2.resize(cv2.imread(img1, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)
    i2 = cv2.resize(cv2.imread(img2, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)

    similarity = 0.0

    if algorithm == 'SIFT':
        # Using OpenCV for feature detection and matching
        sift = cv2.xfeatures2d.SIFT_create()
        k1, d1 = sift.detectAndCompute(i1, None)
        k2, d2 = sift.detectAndCompute(i2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1, d2, k=2)

        for m, n in matches:
            if m.distance < SIFT_RATIO * n.distance:
                similarity += 1.0

        # Custom normalization for better variance in the similarity matrix
        if similarity == len(matches):
            similarity = 1.0
        elif similarity > 1.0:
            similarity = 1.0 - 1.0/similarity
        elif similarity == 1.0:
            similarity = 0.1
        else:
            similarity = 0.0
    elif algorithm == 'CW-SSIM':
        # FOR EXPERIMENTS ONLY!
        # Very slow algorithm - up to 50x times slower than SIFT or SSIM.
        # Optimization using CUDA or Cython code should be explored in the future.
        similarity = pyssim.SSIM(img1).cw_ssim_value(img2)
    elif algorithm == 'SSIM':
        # Default SSIM implementation of Scikit-Image
        similarity = ssim(i1, i2)
    else:
        # Using MSE algorithm with custom normalization
        err = np.sum((i1.astype("float") - i2.astype("float")) ** 2)
        err /= float(i1.shape[0] * i2.shape[1])

        if err > 0.0:
            similarity = MSE_NUMERATOR / err
        else:
            similarity = 1.0

    return similarity

# Fetches all images from the provided directory and calculates the similarity
# value per image pair.
def build_similarity_matrix(images, algorithm='SSIM'):
    num_images = len(images)
    sm = np.zeros(shape=(num_images, num_images), dtype=np.float64)
    np.fill_diagonal(sm, 1.0)

    print("Building the similarity matrix using %s algorithm for %d images" %
          (algorithm, num_images))
    start_total = datetime.datetime.now()

    # Traversing the upper triangle only - transposed matrix will be used
    # later for filling the empty cells.
    k = 0
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            j = j + k
            if i != j and j < sm.shape[1]:
                    print(images[j].endswith((".jpg", ".jpeg", ".png" )))
                    print(images[j])
                    sm[i][j] = 1 - get_image_similarity(images[i], images[j], algorithm=algorithm)
        k += 1

    # Adding the transposed matrix and subtracting the diagonal to obtain
    # the symmetric similarity matrix
    sm = sm + sm.T - np.diag(sm.diagonal())

    end_total = datetime.datetime.now()
    print("Done - total calculation time: %d seconds" % (end_total - start_total).total_seconds())
    return sm

""" Returns a dictionary with the computed performance metrics of the provided cluster.
    Several functions from sklearn.metrics are used to calculate the following:
    * Silhouette Coefficient
      Values near 1.0 indicate that the sample is far away from the neighboring clusters.
      A value of 0.0 indicates that the sample is on or very close to the decision boundary
      between two neighboring clusters and negative values indicate that those samples might
      have been assigned to the wrong cluster.
    * Completeness Score
      A clustering result satisfies completeness if all the data points that are members of a
      given class are elements of the same cluster. Score between 0.0 and 1.0. 1.0 stands for
      perfectly complete labeling.
    * Homogeneity Score
      A clustering result satisfies homogeneity if all of its clusters contain only data points,
      which are members of a single class. 1.0 stands for perfectly homogeneous labeling.
"""
def get_cluster_metrics(X, labels, labels_true=None):
    metrics_dict = dict()
    metrics_dict['Silhouette coefficient'] = metrics.silhouette_score(X,
                                                                      labels,
                                                                      metric='precomputed')
    if labels_true:
        metrics_dict['Completeness score'] = metrics.completeness_score(labels_true, labels)
        metrics_dict['Homogeneity score'] = metrics.homogeneity_score(labels_true, labels)

    return metrics_dict

""" Executes two algorithms for similarity-based clustering:
    * Spectral Clustering
    * Affinity Propagation
    ... and selects the best results according to the clustering performance metrics.
"""
def do_cluster(images, n_clusters, init, algorithm='SIFT', print_metrics=True, labels_true=None):
    #matrix = build_similarity_matrix(images, algorithm=algorithm)
    # read images to matrix
    data = []
    centroids = []

    for i in range(len(images)):
        if(images[i].endswith((".jpg", ".jpeg", ".png" ))):
            aux = np.array(cv2.resize(cv2.imread(images[i], cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)).flatten()
            data.append(aux)
            for name in init:
                if images[i] == name:
                    print(images[i])
                    centroids.append(aux)

    #print("Running %s algorithm for %d images" % (algorithm, len(matrix)))
    #for lst in matrix:
        #print(*lst)
    #clf = DBSCAN(eps=0.6, min_samples=5, metric='precomputed').fit(matrix)
    clf = KMeans(n_clusters, init=centroids).fit(data)
    #sc = SpectralClustering(n_clusters=3, affinity='precomputed').fit(matrix)
    #print("\nPerformance metrics for Spectral Clustering")
    #print("Number of clusters: %d" % len(set(clf.labels_)))
    return clf


def generate_clusters():
    kmeans_list = []
    for i in range(len(num_clusters)):
        if(num_clusters[i] != 1):
            images = os.listdir(dirs[i])
            images = [dirs[i] + '/' + x for x in images]
            kmeans = do_cluster(images,init = inits[i], algorithm='SSIM', print_metrics=True, labels_true=None, n_clusters = num_clusters[i])
            path = "../clusters/clusterClase{}.pkl".format(i)
            pickle.dump(kmeans, open(path, "wb"))
            c = kmeans.labels_
            kmeans_list.append(kmeans)
            for n in range(num_clusters[i]):
                #print("\n --- Images from cluster #%d ---" % n)
                index = np.argwhere(c == n).flatten()
                save_path = f"{dirs[i]}/cluster{n}/"
                for j in index:
                    if (images[j].endswith(('.jpg','.jpeg','png'))):
                        #print(images[j])
                        shutil.copy(images[j], save_path)
  
    return kmeans_list


def get_metrics(images, algorithm='SSIM'):
    num_images = len(images)

    # Traversing the upper triangle only - transposed matrix will be used
    # later for filling the empty cells.
    k = 0
    total  = 0
    for i in range(num_images):
        for j in range(num_images):
            j = j + k
            if i != j and j < num_images and images[i].endswith((".jpg", ".jpeg", ".png" )) and images[j].endswith((".jpg", ".jpeg", ".png" )):
                    total += 1 - get_image_similarity(images[i], images[j], algorithm=algorithm)
                    print(i)
        k += 1

   
    return total/(num_images**2)/2

if __name__ == "__main__":
    generate_clusters()
    clusters = ["../Datasets/Dataset/Femurs/textos/label1/cluster0", "../Datasets/Dataset/Femurs/textos/label1/cluster1", "../Datasets/Dataset/Femurs/textos/label1/cluster2",
                "../Datasets/Dataset/Femurs/textos/label2/cluster0", "../Datasets/Dataset/Femurs/textos/label2/cluster1"]
    values = np.array([])
    for cluster in clusters:
        images = os.listdir(cluster)
        images = [cluster + '/' + x for x in images]
        values = np.append(values, get_metrics(images))
        print("Done")
    
    categorias = ['C1.0', 'C1.1', 'C1.2', 'C2.0', 'C2.1']


    plt.bar(categorias, values)

    # Añadir etiquetas y título
    plt.xlabel('Cluster')
    plt.ylabel('SSIM media')
    plt.title('SSIM media por clusters')

    # Mostrar el gráfico
    plt.show()
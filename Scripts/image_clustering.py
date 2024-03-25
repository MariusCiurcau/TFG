import datetime
import os
import pickle
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
import torch.nn as nn
from PIL import Image
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from model import preprocess

# init_notebook_mode(connected=True)
import scienceplots

plt.style.use(['science', 'no-latex'])

# Constant definitions
SIM_IMAGE_SIZE = (224, 224)
SIFT_RATIO = 0.7
MSE_NUMERATOR = 1000.0
IMAGES_PER_CLUSTER = 2

model_path = "../models/3clases/resnet18_10_3_AO_AQ_MAL"
dirs = ["../Datasets/Dataset/Femurs/textos/label0", "../Datasets/Dataset/Femurs/textos/label1",
        "../Datasets/Dataset/Femurs/textos/label2"]
num_clusters = [1, 3, 2]
cluster_paths = ["../Datasets/Dataset/Femurs/textos/label1/cluster0",
                 "../Datasets/Dataset/Femurs/textos/label1/cluster1",
                 "../Datasets/Dataset/Femurs/textos/label1/cluster2",
                 "../Datasets/Dataset/Femurs/textos/label2/cluster0",
                 "../Datasets/Dataset/Femurs/textos/label2/cluster1"]
inits = [["../Datasets/Dataset/Femurs/textos/label0/c0.jpg"],
         ["../Datasets/Dataset/Femurs/textos/label1/c0.jpg", "../Datasets/Dataset/Femurs/textos/label1/c1.jpg",
          "../Datasets/Dataset/Femurs/textos/label1/c2.jpg"],
         ["../Datasets/Dataset/Femurs/textos/label2/c0.jpg", "../Datasets/Dataset/Femurs/textos/label2/c1.jpg"]]


def get_image_similarity(img1, img2):
    i1 = cv2.resize(cv2.imread(img1, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)
    i2 = cv2.resize(cv2.imread(img2, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)
    return ssim(i1, i2)


# Fetches all images from the provided directory and calculates the similarity
# value per image pair.
def build_similarity_matrix(images):
    num_images = len(images)
    sm = np.zeros(shape=(num_images, num_images), dtype=np.float64)
    np.fill_diagonal(sm, 1.0)

    start_total = datetime.datetime.now()

    # Traversing the upper triangle only - transposed matrix will be used
    # later for filling the empty cells.
    k = 0
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            j = j + k
            if i != j and j < sm.shape[1]:
                print(images[j].endswith((".jpg", ".jpeg", ".png")))
                print(images[j])
                sm[i][j] = 1 - get_image_similarity(images[i], images[j])
        k += 1

    # Adding the transposed matrix and subtracting the diagonal to obtain
    # the symmetric similarity matrix
    sm = sm + sm.T - np.diag(sm.diagonal())

    end_total = datetime.datetime.now()
    print("Done - total calculation time: %d seconds" % (end_total - start_total).total_seconds())
    return sm


def ssim_distance(image1, image2):
    # Compute SSIM between the two images
    similarity = ssim(image1, image2, multichannel=True, data_range=1.0)
    # SSIM ranges from -1 to 1, but distance should be positive, so we return 1 - similarity
    return 1 - similarity


def RMSE(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))


def mse_distance(image1, image2):
    return np.mean((image1 - image2) ** 2)


""" Executes two algorithms for similarity-based clustering:
    * Spectral Clustering
    * Affinity Propagation
    ... and selects the best results according to the clustering performance metrics.
"""


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            *list(model.children())[:-2],  # Remove avgpool and fc layers
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x)



def do_cluster(images, n_clusters, init, distance, use_features=False):
    data = []
    centroids = []

    if use_features:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
        model.load_state_dict(torch.load(model_path))
        model = FeatureExtractor(model)
        model.eval()

    for i in range(len(images)):
        if images[i].endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(images[i])
            input_image = preprocess(image).unsqueeze(0)

            if use_features:
                vector = model(input_image).detach().numpy()[0].flatten().astype(np.double)  # features
            else:
                vector = np.array(cv2.resize(cv2.imread(images[i], cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)).flatten() # imagen completa
            data.append(vector)
            if images[i] in init:
                print(images[i])
                centroids.append(vector)

    if distance == 'Euclidean': # sklearn
        clf = KMeans(n_clusters, init=centroids).fit(data)
        # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, clf.predict(data)))
    else: # pyclustering
        custom_metric = distance_metric(type_metric.USER_DEFINED, func=distance)
        clf = kmeans(data, initial_centers=centroids, metric=custom_metric)  # pyclustering
        clf.process()

    return clf


def generate_clusters(distance, use_features=False):
    kmeans_list = []
    for i in range(len(num_clusters)):
        if num_clusters[i] != 1:
            images = os.listdir(dirs[i])
            images = [dirs[i] + '/' + x for x in images]
            kmeans = do_cluster(images, init=inits[i], n_clusters=num_clusters[i], distance=distance, use_features=use_features)
            path = "../clusters/clusterClase{}.pkl".format(i)
            pickle.dump(kmeans, open(path, "wb"))

            if distance == 'Euclidean': # se utiliza sklearn
                c = kmeans.labels_
            else: # se utiliza pyclustering
                clusters = kmeans.get_clusters() # pyclustering
                c = np.zeros(len(images), dtype=int)
                for cluster_index, cluster in enumerate(clusters):
                    for point_index in cluster:
                        c[point_index] = cluster_index

            kmeans_list.append(kmeans)
            for n in range(num_clusters[i]):
                # print("\n --- Images from cluster #%d ---" % n)
                index = np.argwhere(c == n).flatten()
                save_path = f"{dirs[i]}/cluster{n}/"
                for j in index:
                    if (images[j].endswith(('.jpg', '.jpeg', 'png'))):
                        # print(images[j])
                        shutil.copy(images[j], save_path)
    return kmeans_list


def get_metrics(images):
    num_images = len(images)
    num_pares = 0.
    total = 0.
    for i in range(num_images):
        print(i)
        for j in range(i + 1, num_images):
            num_pares += 1
            total += get_image_similarity(images[i], images[j])
    return total / num_pares


def get_clusters_df():
    print("Graphing clusters...")
    column_names = ['var{}_{}'.format(i, j) for i in range(224) for j in range(224)]
    data = []
    labels = []
    clusters = []
    for cluster_path in cluster_paths:
        parts = cluster_path.split('/')
        label = int(parts[-2][-1])
        cluster = int(parts[-1][-1])
        images = [cluster_path + '/' + x for x in os.listdir(cluster_path) if x.endswith((".jpg", ".jpeg", ".png"))]
        print(label, cluster, len(images))
        for image in images:
            vector = np.array(cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE) / 255.0, SIM_IMAGE_SIZE)).flatten()
            data.append(vector)
            labels.append(label)
            clusters.append(cluster)
    df = pd.DataFrame(data, columns=column_names)
    df['label'] = labels
    df['cluster'] = clusters
    return df


def plot_clusters(df, n_dim=2):
    traces = []
    labels = df['label'].unique().astype(int)
    colorscale = 'Viridis'
    for l in labels:
        clusters = df[df['label'] == l]['cluster'].unique().astype(int)
        for c in clusters:
            cluster = df[(df['label'] == l) & (df['cluster'] == c)]
            if n_dim == 2:
                trace = go.Scatter(
                    x=cluster["pc1"],
                    y=cluster["pc2"],
                    mode="markers",
                    name="Cluster {}.{}".format(l, c),
                    marker=dict(color=l + c, colorscale=colorscale),
                    text=None)
            elif n_dim == 3:
                trace = go.Scatter3d(
                    x=cluster["pc1"],
                    y=cluster["pc2"],
                    z=cluster["pc3"],
                    mode="markers",
                    name="Cluster {}.{}".format(l, c),
                    marker=dict(color=l + c, colorscale=colorscale),
                    text=None)
            traces.append(trace)
    title = f"Visualizing Clusters in {n_dim} Dimensions Using PCA"
    # if n_dim == 2:
    layout = dict(title=title,
                  xaxis=dict(title='PC1', ticklen=5, zeroline=False),
                  yaxis=dict(title='PC2', ticklen=5, zeroline=False)
                  )
    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    # iplot(fig)


def graph_clusters(method='PCA', n_dim=2):
    print("Graphing clusters...")
    df = get_clusters_df()
    columns = df.columns

    df = pd.DataFrame(np.array(df.sample(500)))
    df.columns = columns

    if method == 'PCA':
        pca = PCA(n_components=n_dim)
        pcs = pd.DataFrame(pca.fit_transform(df.drop(["cluster", "label"], axis=1)))
        pcs.columns = ["pc{}".format(x + 1) for x in range(len(pcs.columns))]
        df = pd.concat([df, pcs], axis=1, join="inner")
    plot_clusters(df=df[df["label"] == 1], n_dim=n_dim)
    # elif method == 'T-SNE':

def plot_metrics(metrics, use_features=False, savefig=None):
    clusters = list(metrics.values())[0].keys()
    n_clusters = len(clusters)
    n_distances = len(metrics)

    bar_width = 0.8 / n_distances
    index = np.arange(n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = []
    for i, (dist_name, dist_values) in enumerate(metrics.items()):
        values = list(dist_values.values())
        bars.append(ax.bar(index + i * bar_width, values, bar_width, label=dist_name))

        # Add numeric values on top of each bar
        for j, value in enumerate(values):
            ax.text(index[j] + i * bar_width, value + 0.005, str(round(value, 3)), ha='center')

    ax.set_xlabel('Clusters')
    ax.set_ylabel('Average SSIM')
    if use_features:
        title = 'Average SSIM per cluster using feature extraction'
    else:
        title = 'Average SSIM per cluster without feature extraction'
    ax.set_title(title)
    ax.set_xticks(index + 0.5 * bar_width)
    ax.set_xticklabels(clusters)
    ax.legend()
    if savefig is not None:
        plt.savefig(savefig, dpi=600)
    plt.show()

def image_clustering(distance, use_features=False):
    for dir in cluster_paths:
        for file in os.listdir(dir):
            os.remove(dir + "/" + file)
    classifiers = generate_clusters(distance, use_features=use_features)
    return classifiers

def compute_metrics():
    values = {}
    for cluster_path in cluster_paths:
        images = [cluster_path + '/' + x for x in os.listdir(cluster_path)]
        parts = cluster_path.split('/')
        label = int(parts[-2][-1])
        cluster = int(parts[-1][-1])
        c = f"C{label}.{cluster}"
        values[c] = get_metrics(images)
        print("Done")
    return values


if __name__ == "__main__":
    use_features = True
    distances = ['Euclidean', RMSE]
    metrics = {}
    """
    # Comparación de métricas
    for distance in distances:
        image_clustering(distance, use_features=use_features)
        values = compute_metrics()

        if type(distance) == str:
            metrics[distance] = values
        else:
            metrics[distance.__name__] = values

    print(metrics)
    for dist, values in metrics.items():
        print(dist + ' avg. SSIM: ' + str(np.mean(list(values.values()))))
    plot_metrics(metrics, use_features=use_features, savefig='cluster_metrics_features.png')
    """

    # Métricas de una distancia
    distance = 'Euclidean' # o alguna de las funciones definidas, como RMSE
    image_clustering(distance, use_features=use_features)

    #values = compute_metrics()
    #if type(distance) == str:
    #    metrics[distance] = values
    #else:
    #    metrics[distance.__name__] = values

    #print(metrics)
    #plot_metrics(metrics, use_features=use_features)

    # graph_clusters(n_dim=3)
from math import sqrt
import os.path as osp
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
from scipy.spatial import distance

from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import manhattan_distances
from sklearn_extra.cluster import KMedoids

import torch
import torch.nn.functional as F


def chebyshev_distance(X, Y, **kwargs):
    ans = sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2)
    return ans


def GetDivideMethod(
    mask_tensor: torch.Tensor, areas_num: int = 4
) -> None:
    # 下采样
    original_width, original_height = mask_tensor.shape[1:]
    target_height = mask_tensor.shape[1] // 10
    target_width = mask_tensor.shape[2] // 10

    # 使用 interpolate 函数进行下采样
    downsampled_tensor = F.interpolate(
        mask_tensor.unsqueeze(0),
        size=(target_height, target_width),)

    white_pixels = np.column_stack(np.where(downsampled_tensor > 127))
    print(white_pixels.shape)

    kmeans = KMedoids(
        n_clusters=areas_num,
        metric=chebyshev_distance, random_state=42)  # type: ignore
    kmeans.fit(white_pixels)

    # 各个簇的范围
    clusters = {i: [] for i in range(areas_num)}

    cluster_labels = kmeans.labels_
    for cluster_id, pixel in zip(cluster_labels, white_pixels):
        clusters[cluster_id].append(pixel)

    print(list(len(clusters[i]) for i in range(areas_num)))

    print("A")
    return None

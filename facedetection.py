import os
import cv2
import random
import math
import numpy as np
import cyvlfeat as vlfeat
from scipy import ndimage
import time
from skimage.morphology import watershed
from skimage.feature import peak_local_max

def bgr_to_ycbcr(img):
    y = 0.257 * img[:,:,2] + 0.504 * img[:,:,1] + 0.098 * img[:,:,0] + 16
    cb = 0.148 * img[:,:,2] - 0.291 * img[:,:,1] + 0.439 * img[:,:,0] + 128
    cr = 0.439 * img[:,:,2] - 0.368 * img[:,:,1] - 0.071 * img[:,:,0] + 128
    return y, cb, cr

# src : https://stackoverflow.com/questions/34325879/how-to-efficiently-find-clusters-of-like-elements-in-a-multidimensional-array 
def find_clusters(array):
    clustered = np.empty_like(array)
    unique_vals = np.unique(array)
    cluster_count = 0
    for val in unique_vals:
        labelling, label_count = ndimage.label(array == val)
        for k in range(1, label_count + 1):
            clustered[labelling == k] = cluster_count
            cluster_count += 1
    return clustered, cluster_count


def detectFacesMultiScale(image, scale=1.3, skip=5):

    y, cb, cr = bgr_to_ycbcr(image)


    skinMask = (cr > 130) * (cb > 80) * (y > 80) * (cr < 165) * (cr < 185)
    skinMask = ndimage.binary_erosion(skinMask, structure=np.ones((9,9))).astype(skinMask.dtype)
    skinMask = ndimage.morphology.binary_dilation(skinMask, structure=np.ones((16,16))).astype(skinMask.dtype)
    skinMask = skinMask.astype(float)

    
    clusters, cluster_count = find_clusters(skinMask)

    drawn = image.copy()

    ones = np.ones_like(skinMask, dtype=int)
    cluster_sizes = ndimage.sum(ones, labels=clusters, index=range(cluster_count)).astype(int)
    com = ndimage.center_of_mass(ones, labels=clusters, index=range(cluster_count))

    faces = []

    for i, (size, center) in enumerate(zip(cluster_sizes, com)):
        if size < 250000 and size > 1000:
            truths = clusters==i

            horz = np.sum(truths, axis=0)
            horz= horz.nonzero()
            vert = np.sum(truths,axis=1)
            vert = vert.nonzero()

            x = horz[0][0]
            y = vert[0][0]
            w = horz[0][-1] - x
            h = vert[0][-1] - y

            faces.append((x,y,w,h))
    return faces
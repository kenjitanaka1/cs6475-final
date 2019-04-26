from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.utils import shuffle
import os
import cv2
import random
import math
import numpy as np
import cyvlfeat as vlfeat
from joblib import dump, load
from scipy import ndimage
from sklearn.cluster import AgglomerativeClustering
import sklearn.feature_extraction
import time
from skimage.morphology import watershed
from skimage.feature import peak_local_max

N_REGIONS = 2

def bgr_to_ycbcr(img):
    y = 0.257 * img[:,:,2] + 0.504 * img[:,:,1] + 0.098 * img[:,:,0] + 16
    cb = 0.148 * img[:,:,2] - 0.291 * img[:,:,1] + 0.439 * img[:,:,0] + 128
    cr = 0.439 * img[:,:,2] - 0.368 * img[:,:,1] - 0.071 * img[:,:,0] + 128
    return y, cb, cr

classifier = AdaBoostClassifier(n_estimators=200,algorithm='SAMME.R', random_state=16)

# http://www.robots.ox.ac.uk/~vgg/data3.html
# http://ijettjournal.org/2016/volume-31/number-4/IJETT-V31P236.pdf

def get_img_paths(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file or '.JPG' in file:
                files.append(os.path.join(r, file))
    return files

def trainDetector():
    # # load data
    positivePath = 'data/lfw'
    negativePaths = 'data/HomeObject06Test'
    facePaths = get_img_paths(positivePath)
    negativePaths = get_img_paths(negativePaths)
    faceFeats = []
    cell_size = 6
    for path in facePaths:
        img = cv2.resize(cv2.imread(path),(96,96))
        faceFeats.append(np.reshape(vlfeat.hog.hog(img, cell_size),-1))

    numNegs = math.ceil(len(faceFeats) / len(negativePaths)) * 2
    frameSize = (96,96,3)
    negFeats = []
    for path in negativePaths:
        negImg = cv2.imread(path)
        pts = np.vstack((np.random.randint(negImg.shape[0]-frameSize[0],size=numNegs),
            np.random.randint(negImg.shape[1]-frameSize[1],size=numNegs))).T
        for pt in pts: # select random point
            imgSlice = negImg[pt[0]:pt[0]+frameSize[0], pt[1]:pt[1]+frameSize[1],:]
            negFeats.append(np.reshape(vlfeat.hog.hog(imgSlice,cell_size),-1))
        # extract

    X = faceFeats + negFeats
    y = [True] * len(faceFeats) + [False] * len(negFeats)
    X, y = shuffle(list(X), y, random_state=777)

    print('Dataset size: {}'.format(len(X)))
    classifier.fit(X, y)
    print('Saving model to model.joblib')
    dump(classifier, 'model.joblib') 
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
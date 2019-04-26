from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import fetch_lfw_people
from sklearn.utils import shuffle
import os
import cv2
import random
import math
import numpy as np
import cyvlfeat as vlfeat


classifier = AdaBoostClassifier()

# http://www.robots.ox.ac.uk/~vgg/data3.html

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

    numNegs = math.ceil(len(faceFeats) / len(negativePaths))
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
    X, y = shuffle(X, y, random_state=777)

    print('Dataset size: {}'.format(len(X)))

    classifier.fit(X, y)


    # # the label to predict is the id of the person
    # y = lfw_people.target
    # target_names = lfw_people.target_names
    # n_classes = target_names.shape[0]

    # print("Total dataset size:")
    # print("n_samples: %d" % n_samples)
    # print("n_features: %d" % n_features)
    # print("n_classes: %d" % n_classes)

    # X = lfw_people.data
    # # classifier.fit(X, y)
    
    pass

def detectFacesMultiScale():
    pass
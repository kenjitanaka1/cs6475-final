import math
import cv2
import numpy as np
from time import time
import facedetection
import os

def calc_dispmap(imageL, imageR, boxSize=(9,9), windowWidth = 100):

    imBlurL = cv2.GaussianBlur(cv2.resize(imageL, None, fx=0.5, fy=0.5),(9,9),0).astype(np.int16)
    imBlurR = cv2.GaussianBlur(cv2.resize(imageR, None, fx=0.5, fy=0.5),(9,9),0).astype(np.int16)

    dists = np.zeros((imBlurL.shape[0],imBlurL.shape[1]))
    minDisparities = np.full((imBlurL.shape[0],imBlurL.shape[1]), 256*3)
    kernel = np.ones((9,9),np.float32)/81
    
    for shift in range(int(-windowWidth/2),int(windowWidth/2),2):
        rolled = np.roll(imBlurR, shift=shift, axis=1)
        disparities = np.sum(np.abs(imBlurL - rolled),axis=-1)
        disparities = cv2.filter2D(disparities,-1,kernel)
        mask = (disparities < minDisparities)
        dists[mask] = shift
        minDisparities[mask] = disparities[mask] * 0.99

    dists = cv2.resize(dists, (imageL.shape[1], imageL.shape[0]))
    dists = (np.abs(dists)) / (windowWidth /2.)
    return dists

if __name__ == "__main__":
    # load images: get from webcam instead?
    # note: BGR color, opencv default
    imL = cv2.imread("frame0.png")
    imR = cv2.imread("frame1.png")
    headMask = cv2.imread("headShape.png", 0)
    headSum = np.sum(headMask)

    imgL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

    if not os.path.exists('model.joblib'): # need to train model
        facedetection.trainDetector()
    facesL = facedetection.detectFacesMultiScale(imL, 1.3, 5)

    # face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    # facesL = face_cascade.detectMultiScale(imgL, 1.3, 5)
 
    dists = calc_dispmap(imL, imR)
    facesImage = imL.copy()
    headDists = []
    for (x,y,w,h) in facesL:
        cropped = imgL[y:y+h,x:x+w]
        scaled = cv2.resize(cropped, (96, 96)) / 255.
        scaled *= headMask
        headDists.append(np.sum(scaled)/ headSum)
        cv2.rectangle(facesImage, (x, y), (x + w, y+h), (255, 0, 255), 3)

    cv2.imwrite('faces.png',facesImage)

    diff = dists - headDists[0]
    for headDist in headDists[1:]:
        diffTest = dists - headDist
        mask = diffTest < diff
        diff[mask] = diffTest[mask]
    diff = np.abs(diff)
    diff = cv2.GaussianBlur(diff,(5,5),0) * 2

    # cv2.imshow('diff',diff)
    cv2.imwrite('difference.png', diff * 255)
    # cv2.waitKey(0)
        
    # apply blur
    maxKernel = 14
    depth = diff * 3 # scale it so that all depths are between 0 and 3 for input to logistic fcn
    kernelSize = maxKernel / (1 + np.power(0.25, depth - 2.25)) # logistic function for blurring. 
    kernelSize = (np.floor(kernelSize) * 2 + 1).astype(np.int16) # cast to ints for the gaussian kernel
    # generate the blurred image
    blurredImg = np.zeros(imL.shape).astype(np.uint8)
    for kernel in range(1, maxKernel * 2 + 1, 2):
        blurs = cv2.GaussianBlur(imL,(kernel,kernel),0)
        blurredImg[kernelSize==kernel] = blurs[kernelSize==kernel]
    cv2.imshow('Focuser', blurredImg)
    cv2.imwrite('output.png', blurredImg)
    # cv2.imshow('diff', diff)
    cv2.waitKey(0)
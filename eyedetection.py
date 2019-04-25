import math
import cv2
import numpy as np
from scipy import ndimage
from utils import load_data
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/

# Eye detection and tracking in image with complex background 

# Gaussian mixture model for human skin color and its applications
# in image and video databases

# https://ieeexplore.ieee.org/document/982883

# # face detection thresholds
# # file:///Users/kenjitanaka/Downloads/Face_Detection_Using_Color_Thresholding_and_Eigeni.pdf

# def bgr_to_ycbcr(img):
#     y = 0.257 * img[:,:,2] + 0.504 * img[:,:,1] + 0.098 * img[:,:,0] + 16
#     cb = 0.148 * img[:,:,2] - 0.291 * img[:,:,1] + 0.439 * img[:,:,0] + 128
#     cr = 0.439 * img[:,:,2] - 0.368 * img[:,:,1] - 0.071 * img[:,:,0] + 128
#     return np.stack((y, cb, cr), axis=-1)

# def get_hessian(mat):
#     Ix = cv2.Sobel(mat * 1.,cv2.CV_64F,1,0,ksize=3)
#     Iy = cv2.Sobel(mat * 1.,cv2.CV_64F,0,1,ksize=3)
#     Ixx = cv2.Sobel(Ix,cv2.CV_64F,1,0,ksize=3)
#     Ixy = cv2.Sobel(Ix,cv2.CV_64F,0,1,ksize=3)
#     Iyy = cv2.Sobel(Iy,cv2.CV_64F,0,1,ksize=3)
#     Ixy2 = Ixy * Ixy
#     H = Ixx * Ixy - Ixy2
#     return H

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

    # experimental = sobelize(imageL)
    # dists = cv2.GaussianBlur(dists,(3,3),0).astype(np.int16)

    dists = cv2.resize(dists, (imageL.shape[1], imageL.shape[0]))
    dists = (np.abs(dists)) / (windowWidth /2.)
    return dists

def sobelize(image):
    imBlur = cv2.GaussianBlur(image,(9,9),0).astype(np.int16)

    kernel = np.array([[-1, 0, 1],[-2,0, 2], [-1, 0, 1]])
    filteredA = np.sum(np.abs(cv2.filter2D(imBlur,-1,kernel)),axis=-1)/(256*3.)
    kernel = np.array([[-1, -2, -1],[0,0,0], [1, 2, 1]])
    filteredB = np.sum(np.abs(cv2.filter2D(imBlur,-1,kernel)),axis=-1)/(256*3.)

    filtered = 0.5 * (filteredA + filteredB)
    cv2.imshow('filt',filtered)
    return filtered


def get_my_CNN_model_architecture():
    '''
    The network should accept a 96x96 grayscale image as input, and it should output a vector with 30 entries,
    corresponding to the predicted (horizontal and vertical) locations of 15 facial keypoints.
    '''
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(30))

    return model

# # Load training set
# X_train, y_train = load_data()
# # Setting the CNN architecture
# model = get_my_CNN_model_architecture()
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# hist = model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)
# model.save('my_model.h5')

# dist = lambda x, y : math.sqrt((x[0] - y[0]) **2 + (x[1] - y[1]))
model = load_model('my_model.h5')
# Face cascade to detect faces

if __name__ == "__main__":
    # load images: get from webcam instead?
    # note: BGR color, opencv default
    imL = cv2.imread("frame0.png")
    imR = cv2.imread("frame1.png")
    # imgL = bgr_to_ycbcr(imL)

    imgL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    facesL = face_cascade.detectMultiScale(imgL, 1.3, 5)
    facesR = face_cascade.detectMultiScale(imgR, 1.3, 5)

    # make a copy of the original image to plot detections on
    image_with_detectionsL = imL.copy()
    image_with_detectionsR = imR.copy()

    eyesL = []
    eyesR = []

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in facesL:
        cv2.rectangle(image_with_detectionsL,(x,y),(x+w,y+h),(255,0,0),3)
        cropped = imgL[y:y+h,x:x+w]
        scaled = cv2.resize(cropped, (96, 96)) / 255.
        scaled = scaled.reshape(1,96,96,1)
        kps = model.predict(scaled)
        halfW = w/2
        halfH = h/2
        kp = kps[0]
        kpxL = x + (kp[0] * halfW) + halfW
        kpyL = y + (kp[1] * halfH) + halfH
        kpxR = x + (kp[2] * halfW) + halfW
        kpyR = y + (kp[3] * halfH) + halfH
        eyesL.append((kpxL, kpyL, kpxR, kpyR))
 
        cv2.circle(image_with_detectionsL, (int(kpxL), int(kpyL)), 2, (0,255,0), 3)
        cv2.circle(image_with_detectionsL, (int(kpxR), int(kpyR)), 2, (0,255,0), 3)
    for (x,y,w,h) in facesR:
        cv2.rectangle(image_with_detectionsR,(x,y),(x+w,y+h),(255,0,0),3) 
        cropped = imgR[y:y+h,x:x+w]
        scaled = cv2.resize(cropped, (96, 96)) / 255.
        scaled = scaled.reshape(1,96,96,1)
        kps = model.predict(scaled)
        halfW = w/2
        halfH = h/2
        kp = kps[0]
        kpxL = x + (kp[0] * halfW) + halfW
        kpyL = y + (kp[1] * halfH) + halfH
        kpxR = x + (kp[2] * halfW) + halfW
        kpyR = y + (kp[3] * halfH) + halfH
        eyesR.append((kpxL, kpyL, kpxR, kpyR))

        cv2.circle(image_with_detectionsR, (int(kpxL), int(kpyL)), 2, (0,255,0), 3)
        cv2.circle(image_with_detectionsR, (int(kpxR), int(kpyR)), 2, (0,255,0), 3)
    # cv2.imshow('Focuser',image_with_detectionsL)
    # cv2.waitKey(0)
    # print(eyesL)
    # print(eyesR)

    patchesL = [imL[int(eyeL[1])-4:int(eyeL[1])+4,int(eyeL[0])-4:int(eyeL[0])+4,:].astype(float) for eyeL in eyesL]
    patchesR = [imR[int(eyeR[1])-4:int(eyeR[1])+4,int(eyeR[0])-4:int(eyeR[0])+4,:].astype(float) for eyeR in eyesR]

    dists = np.array([[np.sum(np.abs(patchL - patchR)) for patchR in patchesR] for patchL in patchesL])
    dists /= np.max(dists)
    stacked = np.hstack((image_with_detectionsL, image_with_detectionsR))
    eyeMatch = {}
    dists2 = np.array([[np.sqrt((eyeL[0] - eyeR[0] + eyeL[2] - eyeR[2])**2 + (eyeL[1] - eyeR[1] + eyeL[3] - eyeR[3])**2)  for eyeR in eyesR] for eyeL in eyesL])
    dists2 /= np.max(dists2)
    dists = (dists+dists2)
    distOrder = np.argsort(np.min(dists,axis=-1))

    for j in range(len(distOrder)):
        i = distOrder[j]
        dist = dists[i]
        eyeL = eyesL[i]
        eyeMatch[i] = np.argmin(np.abs(dist))
        eyeR = eyesR[eyeMatch[i]]
        cv2.line(stacked, (int(eyeL[0]), int(eyeL[1])), (int(eyesR[eyeMatch[i]][0] + image_with_detectionsL.shape[1]), int(eyesR[eyeMatch[i]][1])), (255, 255, 0), thickness=1)
        cv2.line(stacked, (int(eyeL[2]), int(eyeL[3])), (int(eyesR[eyeMatch[i]][2] + image_with_detectionsL.shape[1]), int(eyesR[eyeMatch[i]][3])), (0, 255, 255), thickness=1)

    # test2 = np.copy(imL).astype(float) /255.
    # for i in range(len(eyeMatch)):
    #     eyeL = eyesL[i]
    #     eyeR = eyesR[eyeMatch[i]]
        
    #     diffX = (eyeL[0] - eyeR[0] + eyeL[2] - eyeR[2]) / 2
    #     diffY = (eyeL[1] - eyeR[1] + eyeL[3] - eyeR[3]) / 2
    #     # print((diffX, diffY))
    #     M = np.float32([[1,0,round(diffX)],[0,1,round(diffY)]])
    #     rows,cols=imgR.shape
    #     imR2 = cv2.warpAffine(imR,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
    #     test2 = np.copy(imL).astype(float) /255. + imR2.astype(float)/255.
    #     test2 /= 2
    #     # cv2.imshow('test2', test2)
    #     # cv2.waitKey(0)

    diff = calc_dispmap(imL, imR)
    diff -= diff[int(eyesL[0][0]), int(eyesL[0][1])]
    diff = np.abs(diff)
    diff = cv2.GaussianBlur(diff,(5,5),0) * 2
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
    cv2.imshow('diff', diff)
    cv2.waitKey(0)
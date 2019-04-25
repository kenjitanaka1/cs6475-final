import math
import cv2
import numpy as np
from scipy import ndimage
# from utils import load_data
# from keras.models import Sequential, load_model
# from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
# from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
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

def colorDifference(imageL, imageR):
    disparityBGR = imageL.astype(np.int16) - imageR
    return np.linalg.norm(disparityBGR, axis=-1)

def calc_dispmap(imageL, imageR, boxSize=(9,9), windowWidth = 100):
    imBlurL = cv2.GaussianBlur(imageL,(5,5),0)
    imBlurR = cv2.GaussianBlur(imageR,(5,5),0)

    kernel = np.array([[-1, 1],[-1,1]])
    filteredL = np.abs(cv2.filter2D(imBlurL,-1,kernel))
    filteredR = np.abs(cv2.filter2D(imBlurR,-1,kernel))

    dists = np.zeros(imBlurL.shape)

    for shift in range(int(-windowWidth/2),int(windowWidth/2)):
        pass

    # for i in range(0, filteredL.shape[0]):
    #     for j in range(0, filteredL.shape[1]):
    #         tot = filteredL[i,j]
    #         # tot = np.sum(filteredL[i:i+boxSize[0],j:j+boxSize[1]])
    #         if tot > edgeThreshold:
    #             print('{} -> {}'.format((i,j),tot))
    #         else:
    #             filteredL[i,j]=0

    
    cv2.imshow('filt',filteredR)
    cv2.waitKey(0)
    # print('ok')
    # shape = imageL.shape
    # print(shape)
    # dispmap = np.zeros((shape[0]-boxSize[0],shape[1]-boxSize[1]))
    # for i in range(0, dispmap.shape[0], boxSize[1]):
    #     for j in range(0, dispmap.shape[1], boxSize[0]):
    #         patchL = imageL[i:i+boxSize[1],j:j+boxSize[0]]
    #         disps = []
    #         for w in range(int(-windowWidth/2),int(windowWidth/2)):
    #             if j+w > 0 and j+boxSize[0]+w < shape[1]:
    #                 patchR = imageR[i:i+boxSize[1],j+w:j+boxSize[0]+w]
    #                 disps.append(np.sum(np.absolute(patchL - patchR)))
    #             else:
    #                 disps.append(0)
    #         mapped = np.argmin(disps) + int(-windowWidth/2)
    #         dispmap[i:i+boxSize[1],j:j+boxSize[0]] = mapped
    #         print("{} -> {}".format((i,j), mapped))

    # dispmap = dispmap.astype(float) - np.min(dispmap)
    # dispmap /= np.max(dispmap)

    # cv2.imshow('hello', dispmap)
    # cv2.waitKey(0)

            
    

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
# model = load_model('my_model.h5')
# Face cascade to detect faces

if __name__ == "__main__":
    # load images: get from webcam instead?
    # note: BGR color, opencv default
    imL = cv2.imread("IMG_4187.jpg")
    imR = cv2.imread("IMG_4188.jpg")
    imL = cv2.imread("IMG_20190413_125410.jpg")
    imR = cv2.imread("IMG_20190413_125407.jpg")
    imL = cv2.resize(imL, None, fx=0.25, fy=0.25)
    imR = cv2.resize(imR, None, fx=0.25, fy=0.25)
    # imgL = bgr_to_ycbcr(imL)

    imgL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

    im0 = cv2.imread('im0.png')
    im1 = cv2.imread('im1.png')
    img0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    calc_dispmap(img0, img1)

    # face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    # # eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    # facesL = face_cascade.detectMultiScale(imgL, 1.3, 5)
    # facesR = face_cascade.detectMultiScale(imgR, 1.3, 5)

    # # make a copy of the original image to plot detections on
    # image_with_detectionsL = imL.copy()
    # image_with_detectionsR = imR.copy()

    # eyesL = []
    # eyesR = []

    # # loop over the detected faces, mark the image where each face is found
    # for (x,y,w,h) in facesL:
    #     cv2.rectangle(image_with_detectionsL,(x,y),(x+w,y+h),(255,0,0),3)
    #     cropped = imgL[y:y+h,x:x+w]
    #     scaled = cv2.resize(cropped, (96, 96)) / 255.
    #     scaled = scaled.reshape(1,96,96,1)
    #     kps = model.predict(scaled)
    #     halfW = w/2
    #     halfH = h/2
    #     kp = kps[0]
    #     kpxL = x + (kp[0] * halfW) + halfW
    #     kpyL = y + (kp[1] * halfH) + halfH
    #     kpxR = x + (kp[2] * halfW) + halfW
    #     kpyR = y + (kp[3] * halfH) + halfH
    #     eyesL.append((kpxL, kpyL, kpxR, kpyR))
 
    #     cv2.circle(image_with_detectionsL, (int(kpxL), int(kpyL)), 2, (0,255,0), 3)
    #     cv2.circle(image_with_detectionsL, (int(kpxR), int(kpyR)), 2, (0,255,0), 3)
    # for (x,y,w,h) in facesR:
    #     cv2.rectangle(image_with_detectionsR,(x,y),(x+w,y+h),(255,0,0),3) 
    #     cropped = imgR[y:y+h,x:x+w]
    #     scaled = cv2.resize(cropped, (96, 96)) / 255.
    #     scaled = scaled.reshape(1,96,96,1)
    #     kps = model.predict(scaled)
    #     halfW = w/2
    #     halfH = h/2
    #     kp = kps[0]
    #     kpxL = x + (kp[0] * halfW) + halfW
    #     kpyL = y + (kp[1] * halfH) + halfH
    #     kpxR = x + (kp[2] * halfW) + halfW
    #     kpyR = y + (kp[3] * halfH) + halfH
    #     eyesR.append((kpxL, kpyL, kpxR, kpyR))
    #     # ticks = time.time()
    #     # for i in range(100):
    #     #     eyes = eye_cascade.detectMultiScale(cropped)
    #     # print('haar time: {}'.format((time.time() - ticks)/100))
    #     # ticks = time.time()
    #     # for i in range(100):
    #     #     kps = model.predict(scaled)
    #     # print('ml time: {}'.format((time.time() - ticks)/100))
    #     # for (ex,ey,ew,eh) in eyes:
    #     #     cv2.rectangle(image_with_detectionsR,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
    #     cv2.circle(image_with_detectionsR, (int(kpxL), int(kpyL)), 2, (0,255,0), 3)
    #     cv2.circle(image_with_detectionsR, (int(kpxR), int(kpyR)), 2, (0,255,0), 3)
    # # cv2.imshow('Focuser',image_with_detectionsL)
    # # cv2.waitKey(0)
    # # print(eyesL)
    # # print(eyesR)
    # # TODO nonmax supression

    # patchesL = [imL[int(eyeL[1])-4:int(eyeL[1])+4,int(eyeL[0])-4:int(eyeL[0])+4,:].astype(float) for eyeL in eyesL]
    # patchesR = [imR[int(eyeR[1])-4:int(eyeR[1])+4,int(eyeR[0])-4:int(eyeR[0])+4,:].astype(float) for eyeR in eyesR]

    # dists = np.array([[np.sum(np.abs(patchL - patchR)) for patchR in patchesR] for patchL in patchesL])
    # dists /= np.max(dists)
    # stacked = np.hstack((image_with_detectionsL, image_with_detectionsR))
    # eyeMatch = {}
    # dists2 = np.array([[np.sqrt((eyeL[0] - eyeR[0] + eyeL[2] - eyeR[2])**2 + (eyeL[1] - eyeR[1] + eyeL[3] - eyeR[3])**2)  for eyeR in eyesR] for eyeL in eyesL])
    # dists2 /= np.max(dists2)
    # dists = (dists+dists2)
    # distOrder = np.argsort(np.min(dists,axis=-1))

    # for j in range(len(distOrder)):
    #     i = distOrder[j]
    #     dist = dists[i]
    #     eyeL = eyesL[i]
    #     eyeMatch[i] = np.argmin(np.abs(dist))
    #     eyeR = eyesR[eyeMatch[i]]
    #     cv2.line(stacked, (int(eyeL[0]), int(eyeL[1])), (int(eyesR[eyeMatch[i]][0] + image_with_detectionsL.shape[1]), int(eyesR[eyeMatch[i]][1])), (255, 255, 0), thickness=1)
    #     cv2.line(stacked, (int(eyeL[2]), int(eyeL[3])), (int(eyesR[eyeMatch[i]][2] + image_with_detectionsL.shape[1]), int(eyesR[eyeMatch[i]][3])), (0, 255, 255), thickness=1)

    # test2 = np.copy(imL).astype(float) /255.
    # for i in range(len(eyeMatch)):
    #     eyeL = eyesL[i]
    #     eyeR = eyesR[eyeMatch[i]]
    #     # print((int(eyeL[0]), int(eyeL[1])))
    #     # print((int(eyeR[0]), int(eyeR[1])))
        
    #     diffX = (eyeL[0] - eyeR[0] + eyeL[2] - eyeR[2]) / 2
    #     diffY = (eyeL[1] - eyeR[1] + eyeL[3] - eyeR[3]) / 2
    #     # print((diffX, diffY))
    #     M = np.float32([[1,0,round(diffX)],[0,1,round(diffY)]])
    #     rows,cols=imgR.shape
    #     imR2 = cv2.warpAffine(imR,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
    #     test2 = np.copy(imL).astype(float) /255. + imR2.astype(float)/255.
    #     test2 /= 2
    #     cv2.imshow('test2', test2)
    #     cv2.waitKey(0)

    # # # cv2.imshow('Focuser',stacked)
    # # # cv2.waitKey(0)
    # # # for each set of matches:
    # # diffs = []
    # # for i in range(1):#range(len(eyeMatch)):
    # #     eyeL = eyesL[i]
    # #     eyeR = eyesR[eyeMatch[i]]
    # #     # print((int(eyeL[0]), int(eyeL[1])))
    # #     # print((int(eyeR[0]), int(eyeR[1])))
        
    # #     diffX = (eyeL[0] - eyeR[0] + eyeL[2] - eyeR[2]) / 2
    # #     diffY = (eyeL[1] - eyeR[1] + eyeL[3] - eyeR[3]) / 2
    # #     # print((diffX, diffY))
    # #     M = np.float32([[1,0,round(diffX)],[0,1,round(diffY)]])
    # #     rows,cols=imgR.shape
    # #     imgR2 = cv2.warpAffine(imgR,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)

    # #     # imR2 = cv2.warpAffine(imR,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
    # #     # test2 += imR2.astype(float)/255.
    # #     # print(imgL.astype(np.float32).shape)
    # #     # print(imgR2.astype(np.float32).shape)
    # #     imgLfloat = imgL.astype(np.float32)
    # #     diff = np.absolute(imgLfloat - imgR2.astype(np.float32))/imgLfloat
    # #     diffs.append(diff)
    # #     # print(np.max(diff))
    # #     # print(np.min(diff))
    # #     # x,y,w,h = facesL[i]
    # #     # cv2.rectangle(diff,(x,y),(x+w,y+h),(1,1,1),3)
    # #     # cv2.imshow('diff', diff)
    # #     # cv2.waitKey(0)
    # # diff = np.amin(diffs, axis=0)
    # # diff = cv2.GaussianBlur(diff,(5,5),0) # average out the diff
    # # cv2.imshow('diff', diff)
    # # cv2.waitKey(0)
    # # # print(test2.shape)
    # # test2 = 1/(len(eyeMatch) + 1) * test2
    # # cv2.imshow('test2', test2)
    # # cv2.waitKey(0)

    # # # apply blur
    # # maxKernel = 14
    # # depth = diff * 3 # scale it so that all depths are between 0 and 3 for input to logistic fcn
    # # kernelSize = maxKernel / (1 + np.power(0.25, depth - 2.25)) # logistic function for blurring. 
    # # kernelSize = (np.floor(kernelSize) * 2 + 1).astype(np.int16) # cast to ints for the gaussian kernel
    # # # generate the blurred image
    # # blurredImg = np.zeros(imL.shape).astype(np.uint8)
    # # for kernel in range(1, maxKernel * 2 + 1, 2):
    # #     blurs = cv2.GaussianBlur(imL,(kernel,kernel),0)
    # #     blurredImg[kernelSize==kernel] = blurs[kernelSize==kernel]
    # # # cv2.imshow('Focuser', blurredImg)
    # # # cv2.waitKey(0)
    
    # # stacked = np.hstack((cv2.GaussianBlur(imL,(maxKernel*2-1,maxKernel*2-1),0), imL))
    # # stacked = np.hstack((blurredImg, imL))

    # # cv2.imshow('Focuser', stacked)
    # # while(True):
    # #     k = cv2.waitKey(1) & 0xFF
    # #     if k == ord('q'):
    # #         break
    # #     if k == ord('r'):
    # #         # cv2.imshow('Focuser', image_with_detectionsR)
    # #         continue
    
    # # cv2.destroyAllWindows()

    # # cb = imgL[:,:,1]
    # # cr = imgL[:,:,2]

    # # crMask = (140 < cr) * (cr < 165)
    # # cbMask = (135 < cb) * (cb < 185)
    # # skinMask = cbMask * crMask
    # # skinMask = ndimage.binary_erosion(skinMask, structure=np.ones((9,9))).astype(skinMask.dtype)
    # # skinMask = ndimage.morphology.binary_dilation(skinMask, structure=np.ones((10,10))).astype(skinMask.dtype)
    # # H = get_hessian(skinMask)

    # # # Set up the detector with default parameters.
    # # params = cv2.SimpleBlobDetector_Params()
    # # # Change thresholds
    # # params.minThreshold = 4
    # # params.maxThreshold = 20000
    # # params.filterByInertia = True
    # # params.minInertiaRatio = 0.01
    # # detector = cv2.SimpleBlobDetector_create(params)
    # # # Detect blobs.
    # # keypoints = detector.detect(skinMask.astype('uint8') * 255)
    # # cv2.imshow('Focuser', 255 * skinMask.astype('uint8'))
    # # print(keypoints)
    # # # Draw detected blobs as red circles.
    # # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # # im_with_keypoints = cv2.drawKeypoints(imL, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # # Show keypoints
    # # # cv2.imshow("Keypoints", im_with_keypoints)

    # # skinMask = np.repeat(skinMask[:,:,np.newaxis], 3, axis=2)
    # # skinned = skinMask * imL

    # # neural network to recognize iris patterns?
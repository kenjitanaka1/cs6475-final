import cv2
import numpy as np
from scipy import ndimage
import matplotlib
from utils import load_data
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# Eye detection and tracking in image with complex background 

# Gaussian mixture model for human skin color and its applications
# in image and video databases

# https://ieeexplore.ieee.org/document/982883

# face detection thresholds
# file:///Users/kenjitanaka/Downloads/Face_Detection_Using_Color_Thresholding_and_Eigeni.pdf

def bgr_to_ycbcr(img):
    y = 0.257 * img[:,:,2] + 0.504 * img[:,:,1] + 0.098 * img[:,:,0] + 16
    cb = 0.148 * img[:,:,2] - 0.291 * img[:,:,1] + 0.439 * img[:,:,0] + 128
    cr = 0.439 * img[:,:,2] - 0.368 * img[:,:,1] - 0.071 * img[:,:,0] + 128
    return np.stack((y, cb, cr), axis=-1)

def get_hessian(mat):
    Ix = cv2.Sobel(mat * 1.,cv2.CV_64F,1,0,ksize=3)
    Iy = cv2.Sobel(mat * 1.,cv2.CV_64F,0,1,ksize=3)
    Ixx = cv2.Sobel(Ix,cv2.CV_64F,1,0,ksize=3)
    Ixy = cv2.Sobel(Ix,cv2.CV_64F,0,1,ksize=3)
    Iyy = cv2.Sobel(Iy,cv2.CV_64F,0,1,ksize=3)
    Ixy2 = Ixy * Ixy
    H = Ixx * Ixy - Ixy2
    return H

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

# model = load_model('my_model.h5')
# Face cascade to detect faces

if __name__ == "__main__":
    # load images: get from webcam instead?
    # note: BGR color, opencv default
    imL = cv2.imread("IMG_20190413_125407_2.jpg")
    # imL = cv2.resize(imL,(int(imL.shape[1] / 2),int(imL.shape[0] / 2)))
    imR = cv2.imread("IMG_20190413_125410_2.jpg")
    # imgL = bgr_to_ycbcr(imL)

    imgL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    facesL = face_cascade.detectMultiScale(imgL, 1.5, 6)
    facesR = face_cascade.detectMultiScale(imgR, 1.5, 6)

    # make a copy of the original image to plot detections on
    image_with_detectionsL = imL.copy()
    image_with_detectionsR = imR.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in facesL:
        cv2.rectangle(image_with_detectionsL,(x,y),(x+w,y+h),(255,0,0),3) 
    for (x,y,w,h) in facesR:
        cv2.rectangle(image_with_detectionsR,(x,y),(x+w,y+h),(255,0,0),3) 

    # cb = imgL[:,:,1]
    # cr = imgL[:,:,2]

    # crMask = (140 < cr) * (cr < 165)
    # cbMask = (135 < cb) * (cb < 185)
    # skinMask = cbMask * crMask
    # skinMask = ndimage.binary_erosion(skinMask, structure=np.ones((9,9))).astype(skinMask.dtype)
    # skinMask = ndimage.morphology.binary_dilation(skinMask, structure=np.ones((10,10))).astype(skinMask.dtype)
    # H = get_hessian(skinMask)

    # # Set up the detector with default parameters.
    # params = cv2.SimpleBlobDetector_Params()
    # # Change thresholds
    # params.minThreshold = 4
    # params.maxThreshold = 20000
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    # detector = cv2.SimpleBlobDetector_create(params)
    # # Detect blobs.
    # keypoints = detector.detect(skinMask.astype('uint8') * 255)
    # cv2.imshow('Focuser', 255 * skinMask.astype('uint8'))
    # print(keypoints)
    # # Draw detected blobs as red circles.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv2.drawKeypoints(imL, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # Show keypoints
    # # cv2.imshow("Keypoints", im_with_keypoints)

    # skinMask = np.repeat(skinMask[:,:,np.newaxis], 3, axis=2)
    # skinned = skinMask * imL

    # neural network to recognize iris patterns?
    
    cv2.imshow('Focuser', image_with_detectionsL)
    while(True):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            cv2.imshow('Focuser', image_with_detectionsR)
    
    cv2.destroyAllWindows()
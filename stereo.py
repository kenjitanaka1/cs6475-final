import math
import cv2
import numpy as np
#from scipy import ndimage
# from utils import load_data
# from keras.models import Sequential, load_model
# from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
# from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

while(True):
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
stacked = np.hstack((frame0, frame1))
cv2.imwrite('frame0.png', frame0)
cv2.imwrite('frame1.png', frame1)
cv2.imshow('frames', stacked)
cv2.waitKey(0)
# cv2.waitKey()
cap0.release()
cap1.release()
cv2.destroyAllWindows()

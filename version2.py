import cv2
import numpy as np
import matplotlib
from sys import platform as sys_pf
if sys_pf == 'darwin': # source: https://github.com/MTG/sms-tools/issues/36 
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# note : read this paper: https://ieeexplore.ieee.org/document/5565062

maxDisparity = 100
disparityThreshold = 4

maxKernel = 8 # for blurring

window = (3, 5) # for windowing

def colorDifference(imageL, imageR):
    disparityBGR = imageL.astype(np.int16) - imageR
    return np.linalg.norm(disparityBGR, axis=-1)

def calculateDisparityMap(imageL, imageR):
    # padding
    imgR = cv2.copyMakeBorder(imR,0,0,maxDisparity,0,cv2.BORDER_CONSTANT)

    minDisparities = np.full((imL.shape[0], imL.shape[1]), 442) # (441.6 is the maximum possible disparity)
    shifts = np.full((imL.shape[0], imL.shape[1]), -1) # basically we are trying to find disparities that stand out, ignoring flat sheets of color. We will fill those in later.
    # note: max shift = min disparity!

    for i in range(maxDisparity):
        imgL = cv2.copyMakeBorder(imL,0,0,i,maxDisparity - i,cv2.BORDER_CONSTANT)
        disparity = colorDifference(imgR, imgL)
        # compare to minDisparities
        cropped = disparity[:,i:imL.shape[1]+i]
        cropped = cv2.GaussianBlur(cropped, window, 0)

        mask = cropped <= minDisparities
        minDisparities[mask] = cropped[mask]
        shifts[mask] = i

        
    depth = np.ones(shifts.shape) - (shifts / maxDisparity)

    # #### OPENCV Included Version #####
    # imgL = cv2.imread('im0.png',0)
    # imgR = cv2.imread('im1.png',0)
    # stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
    # disparity = stereo.compute(imgL,imgR).astype(np.float32)
    # disparity -= np.min(disparity)
    # disparity *= (1/np.max(disparity))
    # depth = 1-disparity
    # plt.imshow(disparity,'gray')
    # plt.show()
    return depth

# param should be [image, depth_estimation]
def focus(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        img = param[0] # reference to the image from input parameters
        depth = param[1] * 3 # scale it so that all depths are between 0 and 3 for input to logistic fcn
        diffDepth = np.abs(depth - depth[y, x]) # calculate the difference in depth of all the pixels from the reference
        kernelSize = maxKernel / (1 + np.power(0.1, diffDepth - 2)) # logistic function for blurring. 
        kernelSize = (np.floor(kernelSize) * 2 + 1).astype(np.int16) # cast to ints for the gaussian kernel

        # generate the blurred image
        blurredImg = np.zeros(img.shape).astype(np.uint8)
        for kernel in range(1, maxKernel * 2 + 1, 2):
            blurs = cv2.GaussianBlur(img,(kernel,kernel),0)
            blurredImg[kernelSize==kernel] = blurs[kernelSize==kernel]
        cv2.imshow('Focuser', blurredImg)

# Pipeline
if __name__ == "__main__":
    # load images: get from webcam instead?
    # note: BGR color, opencv default
    imL = cv2.imread("im0.png")
    imR = cv2.imread("im1.png")

    depth = calculateDisparityMap(imL, imR)
    plt.imshow(depth)
    plt.show()
    
    cv2.imshow('Focuser', imL)
    cv2.setMouseCallback('Focuser', focus, [imL, depth])

    while(True):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            cv2.imshow('Focuser', imL)
    
    cv2.destroyAllWindows()

    # RGB_imL = cv2.cvtColor(imL, cv2.COLOR_BGR2RGB)
    # RGB_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
    # plt.imshow(RGB_imL)
    # plt.show()
    # imgL = cv2.imread("im0.png", 0)
    # imgR = cv2.imread("im1.png", 0)
    # stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
    # disparity = stereo.compute(imgL,imgR)
    # plt.imshow(disparity,'gray')
    # plt.show()

    # calibrate
    # e = calibrate(imL, imR)
    # # this allows us to calculate the epipolar lines between images
    # # choose correspondences
    # # later change this to use some window size...
    # calculateCorrespondences(imL, imR, e)
    
    # calculate distances
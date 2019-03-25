import cv2
import numpy as np
from matplotlib import pyplot as plt

def colorDifference(color1, color2):
    return np.linalg.norm(color1.astype(np.int16) - color2)

# possible future reference for calibration matrix stuff: http://vision.middlebury.edu/stereo/data/2014/
# takes two images and estimates a fundamental matrix for them. Right now hard coded for the newkuba dataset
def calibrate(img1, img2):
    # calculate correspondences
    u = [[62, 52], [520, 58], [170, 93], [418, 105], [359, 199], [529, 145], [542, 232], [608, 214]]
    v = [[40, 52], [500, 58], [200, 93], [393, 105], [322, 199], [489, 145], [415, 232], [490, 214]]

    assert len(u) == len(v)

    #### 8 Point Algorithm ####
    # TODO normalize
    # TODO try using more points
    # turn into matrix
    y = []
    for i in range(len(u)):
        ui = u[i]
        vi = v[i]
        line = [vi[0] * ui[0], vi[0] * ui[1], vi[0], vi[1] * ui[0], vi[1] * ui[1], vi[1], ui[0], ui[1], 1]
        y.append(line)

    y = np.array(y)
    # solve using SVD
    u, s, vt = np.linalg.svd(y)
    v = vt.T
    f = np.reshape(v[:,-1], (3,3))

    #resolve det(F)=0
    u, s, vt = np.linalg.svd(f)
    s[-1] = 0 # set the smallest singular value to zero
    s = np.diag(s)
    f = u @ s @ vt

    # # unscale
    # f = tb.T @ f @ ta
    return f

def calculateCorrespondences(img1, img2, fundamentalMatrix):

    # u = np.arange(0, img1.shape[1])
    # v = np.arange(0, img1.shape[0])
    # uu, vv = np.meshgrid(u, v, sparse=True)
    # # uv = np.vstack([uu, vv, np.ones(len(uu))])
    # print(uu)

    for v in range(0, img1.shape[0]):
        for u in range(0, img1.shape[1]):
            line = np.matmul(fundamentalMatrix, [u, v, 1])
            slope = line[0] / line[1]
            print(f'{line}, slope: {slope}')
            # start at same point in other image
            intensityL = img1[u, v, :]
            # print(intensityL)
            
            # # check += 10 or something for closest correspondence
            # for delta in range(-9, 10): # range for correspondences # TODO increase
            #     intensityR = img2[u, v + delta, :]
            #     diff = colorDifference(intensityL, intensityR)
            #     print(f'{delta} : {intensityL}, {intensityR}, {diff}')

            # break
        break
    # print("cacl")
    # print(essentialMatrix)

# Pipeline
if __name__ == "__main__":
    # load images: get from webcam instead?
    # note: BGR color, opencv default
    imL = cv2.imread("im0.png")
    imR = cv2.imread("im1.png")

    # imgL = cv2.imread("im0.png", 0)
    # imgR = cv2.imread("im1.png", 0)
    # stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
    # disparity = stereo.compute(imgL,imgR)
    # plt.imshow(disparity,'gray')
    # plt.show()

    # calibrate
    e = calibrate(imL, imR)
    # this allows us to calculate the epipolar lines between images
    # choose correspondences
    # later change this to use some window size...
    calculateCorrespondences(imL, imR, e)
    
    # calculate distances
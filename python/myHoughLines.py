import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    peaks = nonMaxSuppression(H)
    (rhos, thetas) = nLargestIndices(peaks, nLines)

    return (np.array(rhos), np.array(thetas))
    
def nonMaxSuppression(img):
    elem = np.ones(shape=img.shape)
    img1 = np.zeros(img.shape, dtype=np.int32)     # the new image to return
    dilate = cv2.dilate(img, elem)                 # the dilated image

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # this means the pixel was not local maximum, so dilate has different maximum at anchor point
            if dilate[i, j] > img[i, j]:
                img1[i, j] = 0  # so suppress it
            # otherwise, keep it in the img1 image
            else:
                img1[i, j] = dilate[i, j]

    return img1

def nLargestIndices(arr, n):
    flatArr = arr.flatten()
    flatInd = np.argpartition(flatArr, -n)[-n:]     # used to find largest n indices in 1D array

    rhos = []
    thetas = []
    for i in flatInd:
        rhoInd = i // n
        thetaInd = i % n
        rhos.append(rhoInd)
        thetas.append(thetaInd)
    
    return (rhos, thetas)


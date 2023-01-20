import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    peaks = nonMaxSuppression(H)
    (rhos, thetas) = nLargestIndices(peaks, nLines)

    return (rhos, thetas)
    
def nonMaxSuppression(img):
    elem = np.ones(shape=(3, 3))
    dilate = cv2.dilate(img, elem)  # the dilated image
    diff = np.absolute(dilate-img)
    diffInd = np.nonzero(diff)
    img[diffInd] = 0

    return img

def nLargestIndices(arr, n):
    ind = np.argpartition(-arr, n, None)[:n]
    x, y = np.unravel_index(ind, arr.shape)

    return (x, y)

    


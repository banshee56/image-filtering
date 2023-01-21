import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    img = H.copy()                                  # create copy to alter
    peaks = nonMaxSuppression(img)                  # run non maximum suppression
    (rhos, thetas) = nLargestIndices(peaks, nLines) # get the largest n peaks

    return (rhos, thetas)
    
def nonMaxSuppression(img):
    elem = np.ones(shape=(3, 3))    # regular 3x3 kernel to get all neighbors
    dilate = cv2.dilate(img, elem)  # the dilated image, replaced each pixel with max neighbor
    diff = np.absolute(dilate-img)  # diff array has 0s where pixels were local maxima
    diffInd = np.nonzero(diff)      # get indices nonzeros (where pixels were replaced by more intense neighbors)
    img[diffInd] = 0                # suppress non-local maxima pixels

    return img

def nLargestIndices(arr, n):
    ind = np.argpartition(-arr, n, None)[:n]    # the top n indices
    x, y = np.unravel_index(ind, arr.shape)     # separating them to rho and theta indices

    return (x, y)

    


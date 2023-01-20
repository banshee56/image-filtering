import numpy as np
from scipy import signal    # For signal.gaussian function
from myImageFilter import myImageFilter
import cv2

# a function that finds edge intensity and orientation in an image
def myEdgeFilter(img0, sigma):
    # setting up the gaussian kernel
    hsize = 2 * np.ceil(3 * sigma) + 1          # size of kernel
    kernel = signal.gaussian(hsize, std=sigma)  # the 1D gaussian kernel
    h = np.outer(kernel, kernel)                # using outer product to get the 2D kernel
    h = h/h.sum()                               # normalizing the gaussian kernel

    # smoothing the image using convolution
    smoothed = myImageFilter(img0, h)

    # the Sobel filters from class notes
    horizontalSobelFilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    verticalSobelFilter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # convolution with Sobel filters
    imgx = myImageFilter(smoothed, horizontalSobelFilter)   # image gradient in the x direction
    imgy = myImageFilter(smoothed, verticalSobelFilter)     # image gradient in the y direction

    # calculating gradient direction and magnitude matrices/2d arrays
    gradientDirection = np.arctan2(imgy, imgx) * (180/np.pi) 
    gradientMagnitude = np.sqrt((np.square(imgx) + np.square(imgy)))

    # make all angles positive, range = [0, 180)
    gradientDirection[gradientDirection < 0] += 180

    # run magnitude image through non-maximum suppression code
    img1 = nonMaxSuppression(gradientMagnitude, gradientDirection)

    # don't want to consider image boundary as an edge
    img1[0] = 0
    img1[:, 0] = 0
    img1[:, -1] = 0
    img1[-1, :] = 0
    
    return img1


def nonMaxSuppression(gradientMagnitude, gradientDirection):
    # the kernels to use in dilation for each angle
    # these kernels are shaped according to the gradient direction
    elem0 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)     # 0 degree kernel shaped to consider the left and right neigbors
    elem45 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)    # 45 degree kernel considers top right and bottom left neighbors
    elem90 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)    # 90 consider top and bottom neighbors
    elem135 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)   # 135 considers top left and bottom right neighbors

    # dilate the same image using the 4 different kernels
    dilate1 = cv2.dilate(gradientMagnitude, elem0)
    dilate2 = cv2.dilate(gradientMagnitude, elem45)
    dilate3 = cv2.dilate(gradientMagnitude, elem90)
    dilate4 = cv2.dilate(gradientMagnitude, elem135)

    m = 45/2    # if theta =  angle - m, then theta can be quantized to angle

    # get rid of non maxima
    for d in [dilate1, dilate2, dilate3, dilate4]:
        diffArr = np.absolute(d - gradientMagnitude)    # places in image that had the local maxima will now be 0
        diffIndList = np.nonzero(diffArr)               # non zero places are where dilate replaced the pixel with more intense neighbor
        d[diffIndList] = 0                              # get rid of pixels that were changed by dilate


    # stack the dilations on top of one another for easy indexing
    allDilations = np.stack([dilate1, dilate2, dilate3, dilate4])
    indArray = angleToIndex(gradientDirection, m)       # directions indexed where 0degrees->0, 45d->1, 90d->2, 135d->3
    img1 = np.choose(indArray, allDilations)            # using angles as indices to get parts of the stack that correspond to the 'correct' pixel value
    
    return img1


def angleToIndex(g, m):
    # m should be 22.5
    n = np.zeros(shape=g.shape, dtype=int)

    n = np.where((g < (135 + m)) & (g >= m), n, 0)          # change angles in range [0, 22.5) and [157.5, 180] to 0 
    n = np.where((g < m) | (g >= (45 + m)), n, 1)           # change angles in range [22.5, 67.5) to 1
    n = np.where((g < (90 - m)) | (g >= (90 + m)), n, 2)    # change angles in range [67.5, 112.5) to 2
    n = np.where((g < (135 - m)) | (g >= (135 + m)), n, 3)  # change angles in range [112.5, 157.5) to 3
  
    # FOR TESTING ONLY: checking if any angles not indexed
    # print(np.all((n == 0) | (n == 1) | (n == 2) | (n == 3)))
    
    return n

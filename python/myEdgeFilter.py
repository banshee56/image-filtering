import numpy as np
from scipy import signal, ndimage    # For signal.gaussian function
import math
from myImageFilter import myImageFilter
import cv2

# a function that finds edge intensity and orientation in an image
def myEdgeFilter(img0, sigma):
    hsize = 2 * math.ceil(3 * sigma) + 1
    
    kernel = signal.gaussian(hsize, std=sigma)
    h = np.outer(kernel, kernel)
    smoothed = myImageFilter(img0, h)

    # the Sobel filters from class notes
    horizontalSobelFilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    verticalSobelFilter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    imgx = myImageFilter(smoothed, horizontalSobelFilter)   # image gradient in the x direction
    imgy = myImageFilter(smoothed, verticalSobelFilter)     # image gradient in the y direction

    gradientDirection = np.arctan2(imgy, imgx)
    gradientMagnitude = np.hypot(imgx, imgy)

    elem0 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    elem45 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    elem90 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    elem135 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)

    img1 = cv2.dilate(gradientMagnitude, elem0)
    img2 = cv2.dilate(gradientMagnitude, elem45)
    img3 = cv2.dilate(gradientMagnitude, elem90)
    img4 = cv2.dilate(gradientMagnitude, elem135)

    m = 45/2
    new = np.zeros(gradientMagnitude.shape, dtype=np.int32)

    # calculating manually seems to produce better results
    # gradientDirection = np.degrees(gradientDirection)
    gradientDirection = gradientDirection * (180 / math.pi)
    
    for i in range(gradientDirection.shape[0]):
        for j in range(gradientDirection.shape[1]):
            direction = gradientDirection[i, j]

            if direction < 0:
                direction += 180
            
            # angles corresponding to 0 fall in range [-22.5, 22.5)
            if 0-m <= direction < 0+m:
                if img1[i, j] > gradientMagnitude[i, j]:
                    new[i, j] = 0
                else:
                    new[i, j] = img1[i, j]

            elif 45 - m <= direction < 45 + m:
                if img2[i, j] > gradientMagnitude[i, j]:
                    new[i, j] = 0
                else:
                    new[i, j] = img2[i, j]

            elif 90 - m <= direction < 90+m:
                if img3[i, j] > gradientMagnitude[i, j]:
                    new[i, j] = 0
                else:
                    new[i, j] = img3[i, j]

            elif  135 - m <= direction < 180:
                if img4[i, j] > gradientMagnitude[i, j]:
                    new[i, j] = 0
                else:
                    new[i, j] = img4[i, j]
                    
    return new

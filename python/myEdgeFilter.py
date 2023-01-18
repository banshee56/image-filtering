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

    gradientDirection = np.degrees(np.arctan2(imgy, imgx))
    gradientMagnitude = np.hypot(imgx, imgy)

    # the kernels to use in dilation for each angle
    # these kernels are shaped according to the gradient direction
    elem0 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)     # 0 angle kernel shaped to consider the left and right pixels
    elem45 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    elem90 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    elem135 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)

    dilate1 = cv2.dilate(gradientMagnitude, elem0)
    dilate2 = cv2.dilate(gradientMagnitude, elem45)
    dilate3 = cv2.dilate(gradientMagnitude, elem90)
    dilate4 = cv2.dilate(gradientMagnitude, elem135)

    m = 45/2    # if theta =  angle - m, then theta can be quantized to angle
    img1 = np.zeros(gradientMagnitude.shape, dtype=np.int32)     # the new image to return
    
    for i in range(1, gradientDirection.shape[0]-1):
        for j in range(1, gradientDirection.shape[1]-1):
            direction = gradientDirection[i, j]

            # making all angles positive to make quantizing simpler
            if direction < 0:
                direction += 180
            
            # angles corresponding to 0 fall in range [-22.5, 22.5)
            if 0-m <= direction < 0+m:
                # if dilation changed the anchor pixel
                # then pixel at gradientMagnitude[i, j] had more intense neighbor with same gradient direction
                # thus, suppress anchor pixel to 0
                if dilate1[i, j] > gradientMagnitude[i, j]:
                    img1[i, j] = 0
                # otherwise, keep it in the img1 image
                else:
                    img1[i, j] = dilate1[i, j]

            elif 45 - m <= direction < 45 + m:
                if dilate2[i, j] > gradientMagnitude[i, j]:
                    img1[i, j] = 0
                else:
                    img1[i, j] = dilate2[i, j]

            elif 90 - m <= direction < 90+m:
                if dilate3[i, j] > gradientMagnitude[i, j]:
                    img1[i, j] = 0
                else:
                    img1[i, j] = dilate3[i, j]

            elif  135 - m <= direction < 180:
                if dilate4[i, j] > gradientMagnitude[i, j]:
                    img1[i, j] = 0
                else:
                    img1[i, j] = dilate4[i, j]
                    
    return img1

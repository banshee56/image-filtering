import numpy as np
import cv2 

img1 = np.array(np.random.randint(1, 20, size=(180, 240)))
print(img1)

# def nLargestIndices(arr, n):
#     ind = np.argpartition(-arr, n, None)[:n]
#     x, y = np.unravel_index(ind, arr.shape)
#     print(x)
#     print(y)
#     print(arr[x, y])
#     print(type(x))

# nLargestIndices(img1, 4)

def nonMaxSuppression(img):
    elem = np.ones(shape=(3, 3))
    dilate = cv2.dilate(img, elem)  # the dilated image
    diff = np.absolute(dilate-img)
    diffInd = np.nonzero(diff)
    img[diffInd] = 0

    return img

print(nonMaxSuppression(img1))
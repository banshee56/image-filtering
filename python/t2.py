import numpy as np
import cv2 

rhoScale = np.array([0, 2, 4, 6, 8, 10])
rhos = np.random.randint(0, 3, size=5)
thetas =  np.random.randint(0, 5, size=5)

print(rhos)
print(thetas)

h = np.histogram2d(rhos, thetas, rhoScale)
print(h)

# # edgePoints = np.transpose(np.nonzero(img1))
# rhos = np.array([4.3, 8.7, 2.0, 0.4])
# thetas = np.array([1, 4, 2, 6])
# # thetas = np.array([[1, 2, 3], [4, 6, 9], [1, 3, 4] , [7, 8, 2]])
# print(np.histogram2d(rhos, thetas, rhoScale))

# (x, y) = np.unravel_index(edgePoints, img1.shape)

# for pair in edgePoints:
#     p = tuple(pair)
#     print(img1[p])


# rhos =  x*np.cos(thetaScale) + y*np.sin(thetaScale)

# print(thetaScale)
# print(rhos)

# def nLargestIndices(arr, n):
#     ind = np.argpartition(-arr, n, None)[:n]
#     x, y = np.unravel_index(ind, arr.shape)
#     print(x)
#     print(y)
#     print(arr[x, y])
#     print(type(x))

# nLargestIndices(img1, 4)

# def nonMaxSuppression(img):
#     elem = np.ones(shape=(3, 3))
#     dilate = cv2.dilate(img, elem)  # the dilated image
#     diff = np.absolute(dilate-img)
#     diffInd = np.nonzero(diff)
#     img[diffInd] = 0

#     return img

# print(nonMaxSuppression(img1))


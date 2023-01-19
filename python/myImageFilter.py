import numpy as np

def myImageFilter(img0, h):
    # how much to pad each axis by
    padRLength = h.shape[0]//2      # pad by half the number of rows in h
    padCLength = h.shape[1]//2      # pad by half the number of columns in h
    
    # padding the image with 0's
    padded = np.pad(img0, ((padRLength, padRLength), (padCLength, padCLength)), mode='constant')
    img1 = np.zeros_like(img0)      # the filtered image, initialized to 0s

    # we will go through each value/element in the kernel and calculate the element's contribution to the final image, img1
    # we need the following variables for the calculation of the window for each kernel value
    paddedLen0 = padded.shape[0]    # len of padded image on 1 axis
    paddedLen1 = padded.shape[1]
    hLen0 = h.shape[0]              # len of kernel on 1 axis
    hLen1 = h.shape[1]
 
    # go though each value in the kernel, h
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            # the value from the kernel to apply to image
            value = h[i, j]

            # the value is multiplied with image intensities in the range [i, valueRange0] horizontally across padded array
            # and range [j, valueRange0] vertically across array
            valueRange0 = paddedLen0 - hLen0 + i
            valueRange1 = paddedLen1 - hLen1 + j

            # the contribution of that value by multiplying with the original (padded) image
            valueContribution = np.multiply(value, padded[i : valueRange0 + 1, j : valueRange1 + 1])

            # add contribution to final image (which is initialized to 0)
            img1 = np.add(img1, valueContribution)
           
    return img1


""" General utilities for saliency detection.
"""
import cv2
import numpy as np
import math

def saliencyThresh(saliencyMap):
    """ Detect proto-objects from a saliency map using the thresholding technique by
        Hou, 2007.

    Args:
        saliencyMap (array): floating point saliency map to detect object in.
    Returns:
        A floating point image where a pixel is 1 if there is a proto-object detected,
        0 otherwise.
    """
    floatSaliency = np.array(saliencyMap, dtype='float32')
    threshValue = np.mean(floatSaliency) / 3
    retval, thresholded = cv2.threshold(floatSaliency, threshValue, 1, cv2.THRESH_BINARY)
    return thresholded

def centerMap(rows, cols, sigmaFactor = 1./6.):
    """ Returns a center saliency map, which is simply a centered gaussian. In practice
        outperforms many saliency models on its own.

    Args:
        rows, cols: shape of the output map.
        sigmaX, sigmaY: variance parameters to the gaussian.
    Returns:
        A center saliency map.
    """
    sigmaX = cols * sigmaFactor
    sigmaY = rows * sigmaFactor
    sigmaX2 = 2*sigmaX**2
    sigmaY2 = 2*sigmaY**2
    mx = cols / 2
    my = rows / 2
    
    saliency = np.empty([rows, cols], dtype=np.float64)

    for i in range(0,rows):
        for j in range(0,cols):
            immy = float(i) - my
            jmmx = float(j) - mx
            saliency[i,j] = np.exp(-(immy*immy/sigmaY2 +
                                     jmmx*jmmx/sigmaX2))
    return saliency

import numpy as np
import scipy as sp
from scipy import fftpack as fft
import cv2
import math
import os
import os.path
from scipy.ndimage.filters import gaussian_filter

def showScaled(winName, grayscaleImage):
    maxVal = np.amax(grayscaleImage)
    minVal = np.amin(grayscaleImage)
    print winName + ' min: ' + repr(minVal) + ', max: ' + repr(maxVal)
    cv2.imshow(winName, (grayscaleImage - minVal) / (maxVal - minVal))
    cv2.waitKey(0)

def spectralResidualSaliency(grayscaleImage, avgHalfsize = 8, gaussianSigma = 32, maxDim = 500):
    """Computes a saliency map of an image using the spectral residual saliency method
    from Hou, 2007.

    Args:
        grayscaleImage (array): grayscale image to compute a saliency map for.
        avgHalfize (int): half size of the window for the average filter.
        gaussianSigma (number): sigma parameter to the final gaussian filter.
        maxDim (int): maximum size of the largest dimension for the output saliency map.
    Returns:
        A saliency map of the input image.
    """
    # Resize the source image
    newSize = None
    sourceRows, sourceCols = grayscaleImage.shape
    
    if sourceRows > sourceCols:
        newSize = (sourceCols * maxDim / sourceRows, maxDim)
    else:
        newSize = (maxDim, sourceRows * maxDim / sourceCols)
    resizedImage = cv2.resize(grayscaleImage, newSize)
    
    # Compute its Fourier spectrum
    spectrum = fft.fft2(resizedImage)
    
    # apply log scaling to the magnitude
    logSpectrum = np.log(np.absolute(spectrum))
    # get the phase of the spectrum
    phase = np.angle(spectrum)
    # compute the residual
    avgFilterSize = avgHalfsize*2+1
    avgFilterKernel = np.ones([avgFilterSize, avgFilterSize]) / avgFilterSize**2
    avgLogSpectrum = cv2.filter2D(logSpectrum, -1, avgFilterKernel)
    residual = logSpectrum - avgLogSpectrum
    # and from it compute the saliency map directly
    saliencyMap = np.real(fft.ifft2(np.exp(residual + phase*1j))**2)
    filteredMap = gaussian_filter(saliencyMap, gaussianSigma)
    # Normalize to [0;1] range
    minSaliency = np.amin(filteredMap)
    maxSaliency = np.amax(filteredMap)

    return (filteredMap - minSaliency) / (maxSaliency - minSaliency)
    

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
    threshValue = np.mean(floatSaliency)
    retval, thresholded = cv2.threshold(floatSaliency, threshValue, 1, cv2.THRESH_BINARY)
    return thresholded

imageFolder = 'data/background'
imageFilenames = [f for f in os.listdir(imageFolder) 
                  if os.path.isfile(os.path.join(imageFolder, f)) 
                  and f.lower().endswith(('.png', '.jpg', '.gif'))]

for imageFilename in imageFilenames:
    image = cv2.imread(os.path.join(imageFolder, imageFilename), 
                       cv2.CV_LOAD_IMAGE_GRAYSCALE)
    saliencyMap = spectralResidualSaliency(image)
    protoObjects = saliencyThresh(saliencyMap)
    cv2.imshow('original', image)
    cv2.imshow('saliency', saliencyMap)
    cv2.imshow('protoObjects', protoObjects)
    cv2.waitKey(0)

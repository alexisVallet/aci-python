import numpy as np
import scipy as sp
from scipy import fftpack as fft
import cv2
import math

def showScaled(winName, grayscaleImage):
    maxVal = np.amax(grayscaleImage)
    minVal = np.amin(grayscaleImage)
    print winName + ' min: ' + repr(minVal) + ', max: ' + repr(maxVal)
    cv2.imshow(winName, (grayscaleImage - minVal) / (maxVal - minVal))
    cv2.waitKey(0)

def spectralResidualSaliency(grayscaleImage, avgHalfsize = 8, gaussianSigma = 64, maxDim = 500):
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
    saliencyMap = fft.ifft2(np.exp(residual + phase*1j))**2
    # Normalize to [0;1] range
    realMap = np.real(saliencyMap)
    minSaliency = np.amin(realMap)
    maxSaliency = np.amax(realMap)

    return (realMap - minSaliency) / (maxSaliency - minSaliency);

def saliencyThresh(saliencyMap):
    threshValue = 3 * np.mean(saliencyMap)
    return cv2.threshold(saliencyMap, threshValue, 1, cv2.THRESH_BINARY)

saliency = spectralResidualSaliency(cv2.imread('testimage.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE))
showScaled('saliency', saliency)
cv2.imshow('proto-objects', saliencyThresh(saliency))
cv2.waitKey(0)

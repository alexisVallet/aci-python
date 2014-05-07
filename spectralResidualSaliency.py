import numpy as np
import scipy as sp
from scipy import fftpack as fft
import cv2
import math
import os
import os.path
from scipy.ndimage.filters import gaussian_filter
import cvUtils
import principalColorSpace as pcs
import saliency

def colorSRS(image, weights = None, avgHalfsize = 8, gaussianSigma = 32, maxDim = 500):
    """ Computes a saliency map of a color image using spectral residual saliency on
        each layer individually, then combines the results by taking a (weighted)
        average.
    """
    # Treat the case of grayscale image specifically.
    if len(image.shape) < 3 or image.shape[2] == 1:
        saliency = spectralResidualSaliency(image, avgHalfsize, gaussianSigma, maxDim)
        return saliency
    # Set the weights if not set.
    rows, cols, channels = image.shape
    if weights == None:
        weights = [1] * channels
    avgResult = None
    # Compute and combine
    for i in range(0,channels):
        channel = np.array(image[:,:,i], copy=False)
        saliency = spectralResidualSaliency(channel, avgHalfsize, gaussianSigma, maxDim)
        if avgResult == None:
            avgResult = weights[i] * saliency
        else:
            avgResult += weights[i] * saliency
    # Our implementation of SRS happens to normalize each layer to the [0;1] range.
    return avgResult / np.sum(weights)

def spectralResidualSaliency(grayscaleImage, avgHalfsize = 4, gaussianSigma = 16, maxDim = 256):
    """Computes a saliency map of an image using the spectral residual saliency method
    from Hou, 2007.

    Args:
        grayscaleImage (array): grayscale image to compute a saliency map for.
        avgHalfize (int): half size of the window for the average filter.
        gaussianSigma (number): sigma parameter to the final gaussian filter.
        maxDim (int): maximum size of the largest dimension for the output saliency map.
    Returns:
        A saliency map of the input image, and optionally the resized source image.
    """
    # Resize the source image
    newSize = None
    sourceRows, sourceCols = grayscaleImage.shape[0:2]
    
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

if __name__ == "__main__":
    for imageFilename in cvUtils.imagesInFolder('data/background'):
        image = cv2.imread(imageFilename)
        # convert to principal color space, use eigenvalues as layer weights
        pcsImage = pcs.convertToPCS(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), 3)
        saliencyMap = colorSRS(pcsImage)
        rows, cols = saliencyMap.shape[0:2]
        w = 0.7
        center = saliency.centerMap(rows, cols)
        centered = center * w + saliencyMap * (1 - w)
        cv2.imshow('original', image)
        cv2.imshow('saliency', saliencyMap)
        cv2.imshow('center map', center)
        cv2.imshow('centered', centered)
        cv2.waitKey(0)

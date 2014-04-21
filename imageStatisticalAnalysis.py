""" Provides functions for statistical analysis of animation images.
"""
import numpy as np
from scipy import fftpack as fft
from scipy import interpolate as interp
import cv2
import os
import matplotlib.pyplot as plt
import cvUtils
import math

def averageImage(filenames, dimensions = (800, 800)):
    """ Computes the average of a set of grayscale images. Color images are
        automatically converted.
    Args:
        filenames (array): filenames of the images to compute the average of.
        dimensions (tuple): (width, height) dimensions of the output average image. 
            All images will be resized to these dimensions before averageing.
    Returns:
        An average grayscale image of the input images.
    """
    (width, height) = dimensions
    average = np.zeros([width, height])
    for filename in filenames:
        image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        resized = cv2.resize(image, dimensions)
        average += resized
    return average / (len(filenames) * 255)

def averageLogSpectrum(filenames, nbFreq = 128, nbAngles = 128):
    avgMagSpectrum = np.zeros([nbFreq])

    for filename in filenames:
        print 'processing ' + filename
        image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        avgOrientSpectrum = averageSpectrumOverOrientations(image, nbFreq, nbAngles)
        avgMagSpectrum += avgOrientSpectrum
    return np.log(avgMagSpectrum / len(filenames))

def averageSpectrumOverOrientations(grayscaleImage, nbFreq = 128, nbAngles = 128):
    # The mathematical reasoning behind the code relies on geometrical insight not
    # easily conveyed in source code comments. Please see the soon to be published
    # blog article about this. TODO: add url when blog post is published.
    # First compute the 2D DFT of the image
    if nbFreq < 2:
        raise ValueError("nbFreq should be at least 2")
    fourierTransform = np.absolute(fft.fftshift(fft.fft2(grayscaleImage)))
    rows, cols = fourierTransform.shape
    # for linear interpolation on the fourier spectrum
    linerp2 = interp.RectBivariateSpline(range(0,rows),range(0,cols),
                                         fourierTransform, kx=1, ky=1)
    hRows, hCols = (rows/2,cols/2)
    # putting the origin in the middle
    centerLerp2 = lambda x, y: linerp2(x + hRows, y + hCols)
    orientationSum = np.zeros([nbFreq])
    # Iterate over orientations
    for j in range(0,nbAngles):
        angle = j * math.pi / nbAngles
        angleCos = math.cos(angle)
        angleSin = math.sin(angle)
        orientation = np.empty([nbFreq])
        # then over frequencies
        for i in range(0,nbFreq):
            freq = i * rows / (2 * (nbFreq - 1))
            orientation[i] = centerLerp2(freq * angleCos, freq * angleSin)
        orientationSum += orientation
    return orientationSum / nbAngles

if __name__ == "__main__":
    imageFolder = 'data/background'
    imageFilenames = [os.path.join(imageFolder, f) for f in sorted(os.listdir(imageFolder), key=str.lower)
                      if os.path.isfile(os.path.join(imageFolder, f)) 
                      and f.lower().endswith(('.png', '.jpg', '.gif'))]

    plt.plot(averageLogSpectrum(imageFilenames))
    plt.ylabel('log spectrum intensity')
    plt.xlabel('frequency')
    plt.show()

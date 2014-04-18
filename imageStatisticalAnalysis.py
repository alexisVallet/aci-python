""" Provides functions for statistical analysis of animation images.
"""
import numpy as np
from scipy import fftpack as fft
import cv2
import os
import matplotlib.pyplot as plt
import cvUtils

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

def averageLogSpectrum(filenames, dimensions = (800, 800)):
    (drows, dcols) = dimensions
    avgLogSpectrum = np.zeros([int(drows * dcols / 2)])

    for filename in filenames:
        print 'processing ' + filename
        image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        resized = cv2.resize(image, dimensions)
        avgLogSpectrum += np.absolute(logSpectrum(resized))

    return avgLogSpectrum / len(filenames)

def logSpectrum(image):
    spectrumCol = fft.fft(image.flatten('F'))
    spectrumRow = fft.fft(image.flatten('C'))
    avgMagnitudeSpectrum = (np.absolute(spectrumCol) + np.absolute(spectrumRow)) / 2
    n = len(avgMagnitudeSpectrum)
    return np.log(1 + avgMagnitudeSpectrum)[0:int(n/2)]

if __name__ == "__main__":
    imageFolder = 'data/background'
    imageFilenames = [os.path.join(imageFolder, f) for f in sorted(os.listdir(imageFolder), key=str.lower)
                      if os.path.isfile(os.path.join(imageFolder, f)) 
                      and f.lower().endswith(('.png', '.jpg', '.gif'))]
    spectrum = averageLogSpectrum(imageFilenames, (32, 32))
    plt.plot(spectrum)
    plt.ylabel('log spectrum intensity')
    plt.xlabel('frequency')
    plt.show()

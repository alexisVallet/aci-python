import sklearn.decomposition as decomp
import numpy as np
import cv2
import cvUtils
import os.path

def convertToPrincipalColor(colorImage):
    """ Converts a color image to grayscale using the principal components of all pixels
        in the original color space.
    Args:
        colorImage (image): n-channel image in some color space.
    Returns:
        A grayscale image representing the original image along its principal component,
        normalized to the [0;1] range.
    """
    rows, cols, channels = colorImage.shape
    # converts the pixel to a sample matrix usable by the PCA function
    pixels = np.reshape(colorImage, [rows * cols, channels])
    pca = decomp.PCA(n_components = 1)
    newPixels = pca.fit_transform(pixels)
    minVal = np.amin(newPixels)
    maxVal = np.amax(newPixels)

    return np.reshape((newPixels - minVal) / (maxVal - minVal), [rows, cols])

if __name__ == "__main__":
    for filename in cvUtils.imagesInFolder(os.path.join('data', 'background')):
        image = cv2.imread(filename)
        labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        cv2.imshow('source', image)
        cv2.imshow('opencv grayscale', cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        cv2.imshow('PCA BGR', convertToPrincipalColor(image))
        cv2.imshow('PCA Lab', convertToPrincipalColor(labImage))
        cv2.waitKey(0)

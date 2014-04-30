import sklearn.decomposition as decomp
import numpy as np
import cv2
import cvUtils
import os.path

def convertToPCS(colorImage, outputChannels = 3):
    """ Converts a color image to the color space spanned by the principal components of
        its pixel values.
    Args:
        colorImage (image): n-channel image in some color space.
        outputChannel (int): number of channels in the output image.
    Returns:
        An image representing the original image along its principal components,
        and the corresponding eigenvalues of the covariance matrix of pixel values
        in the image - useful for weighting purposes.
    """
    rows, cols, channels = colorImage.shape
    # converts the pixel to a sample matrix usable by the PCA function
    pixels = np.reshape(colorImage, [rows * cols, channels])
    # Find out the eigenvalues of the covariance matrix
    covariance = np.cov(pixels, rowvar = 0)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    # Compute the actual PCA
    pca = decomp.PCA(n_components = outputChannels)
    newPixels = pca.fit_transform(pixels)
    pcsImage = np.reshape(newPixels, [rows, cols, outputChannels])
    normalized = np.empty_like(pcsImage)
    # normalize each layer to the [0;1] range
    for i in range(0, outputChannels):
        minVal = np.amin(pcsImage[:,:,i])
        maxVal = np.amax(pcsImage[:,:,i])
        normalized[:,:,i] = (pcsImage[:,:,i] - minVal) * 255 / (maxVal - minVal)
    # special case for 1 channel cause np kind of suck at it
    if outputChannels == 1:
        normalized.reshape([rows, cols])
    return (normalized.astype(np.float32), eigenvalues)

if __name__ == "__main__":
    for filename in cvUtils.imagesInFolder(os.path.join('data', 'background')):
        image = cv2.imread(filename)
        labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pcsImage = convertToPCS(labImage, 3)
        print pcsImage.shape
        cv2.imshow('bgr', image)
        cv2.imshow('lab', labImage)
        cv2.imshow('lab pcs', pcsImage)
        cv2.waitKey(0)

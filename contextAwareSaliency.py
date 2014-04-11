import numpy as np
import scipy as sp
from scipy import spatial
import cv2
import math

def multiScaleSaliency(bgrImage, patchHalfSize = 7, scales = [1, 0.8, 0.5, 0.3], maxDim = 250, numberOfNeighbors = 64):
    """Computes the saliency map for an image using multi-scale saliency
    from Zelnik-Manor, 2012.
    
    Args:
        bgrImage (array): BGR image to compute the saliency map from.
        patchHalfSizes (array): array of ints of patch half sizes to consider.
            The patch will be centered at a given pixel, so if s is the half
            patch size the patch is a (2s+1)x(2s+1) square.
        scales (array): array of floats providing scales to examine.
        maxDim (int): maximum dimension size of the returned saliency map.
        numberOfNeighbors (int): number of neighbors to consider for each patch in the
            image.
    Returns:
        A saliency map of the input image.
    """
    # First resize the source image and convert it to CIE L*a*b* color space.
    newSize = None
    (sourceRows, sourceCols, nbChannels) = bgrImage.shape
    if sourceRows > sourceCols:
        newSize = (sourceCols * maxDim / sourceRows, maxDim)
    else:
        newSize = (maxDim, sourceRows * maxDim / sourceCols)
    resizedBgrImage = cv2.resize(bgrImage, newSize)
    resizedLabImage = cv2.cvtColor(resizedBgrImage, cv2.COLOR_BGR2LAB)
    
    cv2.imshow('resized and converted', resizedLabImage)
    cv2.waitKey(0)

    # Then compute all patches at all scales, resizing them to the original scale
    # so euclidean distance between patches of different scales is still meaningful.

    # Compute the total number of patches, by summing it for each scale considered.
    (newCols, newRows) = newSize
    numberOfPatches = 0

    halfSizes = map(lambda scale: int(round(scale * patchHalfSize)), scales)

    for hs in halfSizes:
        # We don't take patches with pixels outside of the image, so by drawing
        # the image it is geometrically clear that we have a total of
        # (newRows - 2*hs)*(newCols - 2*hs) pixels to consider and therefore as many
        # patches.
        numberOfPatches += (newRows - 2*hs)*(newCols - 2*hs)
    # The number of patches is the number of rows of our patches array. The number of 
    # columns is the number of pixels in one patch ((2*patchHalfSize + 1)**2) times 
    # the number of channels of an L*a*b* image (3) .
    patches = np.empty([numberOfPatches, 3*((2*patchHalfSize + 1)**2)])
    maxSize = (2*patchHalfSize + 1, 2*patchHalfSize + 1)
    currentRow = 0

    # Resize and flatten each patch into the patches matrix.
    for hs in halfSizes:
        for i in range(hs, newRows - hs - 1):
            for j in range(hs, newCols - hs - 1):
                patch = getPatchAt(resizedLabImage, hs, i, j)
                resizedPatch = cv2.resize(patch, maxSize)
                patches[currentRow] = resizedPatch.flatten()
                currentRow += 1
    # For each normal scale patch, compute its K nearest neighbors among all the scaled
    # and unscaled patches previously computed, and from these compute the pixel
    # saliency.
    saliencyMap = np.zeros([newRows, newCols]o)
    patchesKDTree = spatial.KDTree(patches)
    maxDist = np.linalg.norm(np.array([255, 255, 255]))

    for i in range(patchHalfSize, newRows - patchHalfSize - 1):
        for j in range(patchHalfSize, newCols - patchHalfSize - 1):
            patch = getPatchAt(resizedLabImage, patchHalfSize, i, j)
            # Actually looking up the K + 1 neighbors since the nearest is going to
            # be itself - with a distance of 0, so we don't even have to discard it.
            distances, neighbors = patchesKDTree.query(patch.flatten(), k = numberOfNeighbors + 1)
            # normalize distances to the [0;1] range before computing saliency
            normalizedDistances = distances / maxDist
            saliencyMap[i,j] = 1 - math.exp(-np.sum(distances) / numberOfNeighbors)
    return saliencyMap

def getPatchAt(image, hs, row, col):
    """Cuts a patch out of an image centered at a specific pixel.
    Args:
        image (array): image to cut a patch from.
        hs (int): half size of the patch.
        row (int): row of the pixel at the center of the patch.
        col (int): column of the pixel at the center of the patch.
    """
    return image[row-hs:row+hs+1, col-hs:col+hs+1]

saliency = multiScaleSaliency(cv2.imread('testimage.jpg'), scales = [1], maxDim = 100, numberOfNeighbors = 32)
cv2.imshow('saliency', saliency)
cv2.waitKey(0)

import os
import cv2
import spectralResidualSaliency as srs
import saliency
import cvUtils
import numpy as np
from sklearn import metrics
import principalColorSpace as pcs

def grabcutSaliencyThresh(saliencyMap):
    """ Computes a mask for use with cv2.grabCut from a saliency map by doing a 4-tier
        thresholding.
    """
    # Compute relevant stats about the saliency map first
    meanVal = np.mean(saliencyMap)
    rows, cols = saliencyMap.shape[0:2]
    gcMask = np.empty([rows, cols], dtype=np.uint8)
    # Compute thresholds from the stats
    x2 = meanVal / 3
    
    for i in range(0,rows):
        for j in range(0,cols):
            saliency = saliencyMap[i,j]
            if saliency < x2:
                gcMask[i,j] = cv2.GC_PR_BGD
            else:
                gcMask[i,j] = cv2.GC_PR_FGD

    return gcMask

def grabcutMaskImage(gcMask):
    rows, cols = gcMask.shape[0:2]
    image = np.empty([rows, cols])
    colorMap = { cv2.GC_BGD : 0,
                 cv2.GC_PR_BGD : 0.33,
                 cv2.GC_PR_FGD : 0.66,
                 cv2.GC_FGD : 1 }

    for i in range(0,rows):
        for j in range(0,cols):
            image[i,j] = colorMap[gcMask[i,j]]
    return image

def srsGrabcutBgRemoval(bgrImage, centerPrior = False, centerFactor = 0.7, nbComp = 2,
                        convertToLab = False, maxDim = 500, usePCS = True):
    """ Runs background removal on an image using a combination of spectral residual
        saliency and grabcut. Parametrized to give good results on animation images.
    Args:
        bgrImage (array): bgr image to remove the background from.
    Returns:
        A mask with 1 for foreground and 0 for background, of the same size as the
        source image.
    """
    # in case centerFactor is 1, don't actually run srs :p
    saliencyMap = None
    originalRows, originalCols = bgrImage.shape[0:2]
    rows, cols = cvUtils.maxDimSize(originalRows, originalCols, maxDim)
                  
    if centerPrior and centerFactor == 1:
        saliencyMap = saliency.centerMap(rows, cols)
    else:
        # Convert image to principal color space, apply saliency detection to each 
        # channel, and combine the results using mean weighted by corresponding 
        # eigenvalues.
        cvtImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2LAB) if convertToLab else bgrImage
        if usePCS:
            image, eigenvalues = pcs.convertToPCS(cvtImage, nbComp)
            saliencyMap = srs.colorSRS(image, weights=np.sqrt(eigenvalues), maxDim=maxDim)
        else:
            saliencyMap = srs.colorSRS(cvtImage, maxDim=maxDim)
        # If user chose so, use the center prior heuristic
        if centerPrior:
            center = saliency.centerMap(saliencyMap.shape[0], saliencyMap.shape[1])
            saliencyMap = centerFactor * center + (1 - centerFactor) * saliencyMap
    # generate a mask for the objects in the image.
    charMask = grabcutSaliencyThresh(saliencyMap)
    # Run grabcut to enhance the result
    bgModel, fgModel = [None] * 2
    colorResized = cv2.resize(bgrImage, (cols, rows))
    cv2.grabCut(colorResized, charMask, None, bgModel, fgModel, 1)
    # Only keep the largest connected component, we'll assume it's the character
    cvMaskBool = np.equal(charMask, np.ones([rows, cols]) * cv2.GC_PR_FGD)
    cvMask = cvMaskBool.astype(np.float)
    segmentation2, background2 = cvUtils.connectedComponents(cvMask)
    finalMask = cvUtils.generateMask(segmentation2, segmentation2.getLargestObject())
    # Resize the mask to the original image size.

    return cv2.resize(finalMask, (originalCols, originalRows))

# launched the algorithm in a variety of configurations
if __name__ == "__main__":
    inputFolder = os.path.join('data', 'background')

    for filename in cvUtils.imagesInFolder(inputFolder):
        print 'processing ' + filename
        image = cv2.imread(filename)
        filename = os.path.splitext(os.path.basename(filename))[0] + '.png'
        rootFolder = 'data'

        for colorSpace in ['BGR', 'LAB']:
            for centerPrior in [True, False]:
                for nbComp in [3]:
                    mask = srsGrabcutBgRemoval(image,
                                               centerPrior = centerPrior,
                                               nbComp = nbComp,
                                               convertToLab = (colorSpace == 'LAB'),
                                               usePCS = False)
                    folder = os.path.join(
                        rootFolder,
                        colorSpace + '_' + repr(nbComp) 
                        + ('_center' if centerPrior else '')
                        + '_noPCS')
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    print os.path.join(folder, filename)
                    cv2.imwrite(os.path.join(folder, filename), mask * 255)

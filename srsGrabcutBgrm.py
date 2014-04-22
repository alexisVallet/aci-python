import os
import cv2
import spectralResidualSaliency as srs
import cvUtils
import numpy as np
from sklearn import metrics
import principalColorSpace as pcs

def bgrmRandMeasure(mask1, mask2):
    """ Computes the rand measure between 2 background removals.
    Args:
        mask1 (array): binary mask for the first background removal.
        mask2 (array): binary mask for the second background removal.
    Returns:
        A number between -1 and 1, with the larger number meaning the closest removals.
    """
    return metrics.adjusted_rand_score(mask1.flatten('C'), mask2.flatten('C'))

def srsGrabcutBgRemoval(bgrImage):
    """ Runs background removal on an image using a combination of spectral residual
        saliency and grabcut. Parametrized to give good results on animation images.
    Args:
        bgrImage (array): bgr image to remove the background from.
    Returns:
        A mask with 1 for foreground and 0 for background, of the same size as the
        source image.
    """
    image = pcs.convertToPrincipalColor(cv2.cvtColor(bgrImage, cv2.COLOR_BGR2LAB))
    saliencyMap, resized = srs.spectralResidualSaliency(image)
    protoObjects = srs.saliencyThresh(saliencyMap)
    segmentation, background = cvUtils.connectedComponents(protoObjects)
    # generate a mask for the objects in the image.
    charMask = segmentation.foregroundMask(fgVal=cv2.GC_PR_FGD, bgVal=cv2.GC_PR_BGD)
    # Run grabcut to enhance the result
    bgModel, fgModel = [None] * 2
    rows, cols = resized.shape
    colorResized = cv2.resize(bgrImage, (cols, rows))
    cv2.grabCut(colorResized, charMask, None, bgModel, fgModel, 1)
    # Only keep the largest connected component, we'll assume it's the character
    cvMaskBool = np.equal(charMask, np.ones([rows, cols]) * cv2.GC_PR_FGD)
    cvMask = cvMaskBool.astype(np.float)
    segmentation2, background2 = cvUtils.connectedComponents(cvMask)
    finalMask = cvUtils.generateMask(segmentation2, segmentation2.getLargestObject())
    # Resize the mask to the original image size.
    originalRows, originalCols = bgrImage.shape[0:2]

    return cv2.resize(finalMask, (originalCols, originalRows))

if __name__ == "__main__":
    inputFolder = os.path.join('data', 'background')
    outputFolder = os.path.join('data', 'srs_grabcut_bgrm')
    for filename in cvUtils.imagesInFolder(inputFolder):
        print 'processing ' + filename
        image = cv2.imread(filename)
        mask = srsGrabcutBgRemoval(image)
        cv2.imwrite(os.path.join(outputFolder, os.path.splitext(filename)[0] + '.png'),
                    cvUtils.maskAsAlpha(image, mask))

import os
import cv2
import spectralResidualSaliency as srs
import saliency
import cvUtils
import numpy as np
from sklearn import metrics
import principalColorSpace as pcs
import sys

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

def srsSaliencyMap(bgrImage, nbComp = 1, convertToLab = False, maxDim = 256, 
                   usePCS = True):
    saliencyMap = None
    originalRows, originalCols = bgrImage.shape[0:2]
    rows, cols = cvUtils.maxDimSize(originalRows, originalCols, maxDim)
                  
    # Convert image to principal color space, apply saliency detection to each 
    # channel, and combine the results using mean weighted by corresponding 
    # eigenvalues.
    bgrImage = bgrImage.astype(np.float32)/255
    cvtImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2LAB) if convertToLab else bgrImage
    
    if usePCS:
        image, eigenvalues = pcs.convertToPCS(cvtImage, nbComp)
        saliencyMap = srs.colorSRS(image, weights=np.sqrt(eigenvalues), maxDim=maxDim)
    else:
        saliencyMap = srs.colorSRS(cvtImage, maxDim=maxDim)
    return saliencyMap

def addCenter(saliencyMap, centerSigma, centerFactor):
    center = saliency.centerMap(saliencyMap.shape[0], saliencyMap.shape[1], centerSigma)
    saliencyMap = centerFactor * center + (1 - centerFactor) * saliencyMap

    return saliencyMap

def grabcutBgrm(bgrImage, saliencyMap):
    originalRows, originalCols = bgrImage.shape[0:2]
    rows, cols = saliencyMap.shape[0:2]
    # generate a mask for the objects in the image.
    charMask = grabcutSaliencyThresh(saliencyMap)
    # Run grabcut to enhance the result
    bgModel, fgModel = [None] * 2
    colorResized = (cv2.resize(bgrImage, (cols, rows))*255).astype(np.uint8)
    cv2.grabCut(colorResized, charMask, None, bgModel, fgModel, 1)
    # Only keep the largest connected component, we'll assume it's the character
    cvMaskBool = np.equal(charMask, np.ones([rows, cols]) * cv2.GC_PR_FGD)
    cvMask = cvMaskBool.astype(np.float)
    segmentation2, background2 = cvUtils.connectedComponents(cvMask)
    finalMask = cvUtils.generateMask(segmentation2, segmentation2.getLargestObject())

    # Resize the mask to the original image size.
    return cv2.resize(finalMask, (originalCols, originalRows))

def srsGrabcutBgRemoval(bgrImage, centerPrior = False, centerFactor = 0.7, 
                        centerSigma = 1./6., nbComp = 1,
                        convertToLab = False, maxDim = 256, usePCS = True):
    """ Runs background removal on an image using a combination of spectral residual
        saliency and grabcut. Parametrized to give good results on animation images.
    Args:
        bgrImage (array): bgr image to remove the background from.
    Returns:
        A mask with 1 for foreground and 0 for background, of the same size as the
        source image.
    """
    saliencyMap = srsSaliencyMap(bgrImage, nbComp, convertToLab, maxDim, usePCS)
    if centerPrior:
        saliencyMap = addCenter(saliencyMap, centerSigma, centerFactor)

    return grabcutBgrm(bgrImage, saliencyMap)

# launched the algorithm in a variety of configurations
if __name__ == "__main__":
    inputFolder = os.path.join('data', 'background')
    saliencyFolder = os.path.join('data', 'saliency_maps')
    bgrmFolder = os.path.join('data', 'background_removals')

    # Load all the images in memory because we have a lot of it
    imagesWithNames = lambda folderName, flags: map(lambda filename: (cv2.imread(filename, flags),
                                                                      os.path.splitext(os.path.basename(filename))[0]), 
                                                    cvUtils.imagesInFolder(inputFolder))
    images = imagesWithNames(inputFolder, 1)
    folderName = lambda usePCS, nbComp, convertToLab: os.path.join(saliencyFolder,
                                                                   ('LAB' if convertToLab else 'RGB') + 
                                                                   '_' + repr(nbComp) + 
                                                                   ('' if usePCS else '_noPCS'))

    print "Computing saliency maps..."
    # Compute raw SRS saliency maps and write them to disk if they don't already exist
    for (image, stem) in images:
        for usePCS in [True,False]:
            for nbComp in [1,2,3] if usePCS else [3]:
                for convertToLab in [True,False]:
                    outputFolder = folderName(usePCS, nbComp, convertToLab)
                    outputFilename = os.path.join(outputFolder,
                                                  stem + '.png')
 
                    saliencyMap = srsSaliencyMap(image, usePCS = usePCS,
                                                 convertToLab = convertToLab,
                                                 nbComp = nbComp)
                    cv2.imwrite(outputFilename, saliencyMap * 255)

    sigmas = np.linspace(0.2,1,5).tolist()
    centerFactors = np.linspace(0,1,6).tolist()

    print "Computing centered saliency maps..."
    # Compute saliency maps with centers from these maps
    for usePCS in [True,False]:
        for nbComp in [1,2,3] if usePCS else [3]:
            for convertToLab in [True,False]:
                # load all the maps from one folder in memory just cause
                folder = folderName(usePCS, nbComp, convertToLab)
                saliencyMaps = imagesWithNames(folder, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                for (rawMap, stem) in saliencyMaps:
                    saliencyMap = rawMap.astype(np.float32) / 255
                    for sigma in sigmas:
                        for centerFactor in centerFactors:
                            outFolder = os.path.join(folder + '_center',
                                                     'sigma_' + ("%.1f" % sigma),
                                                     'centerfactor_' + 
                                                     ("%.1f" % centerFactor))
                            try:
                                os.makedirs(outFolder)
                            except:
                                pass
                            outFile = os.path.join(outFolder, stem + '.png')
                            cv2.imwrite(outFile, addCenter(saliencyMap, sigma,
                                                           centerFactor))


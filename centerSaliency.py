""" 'Braindead' center map saliency detection with grabcut. To see whether adding
    spectral residual saliency actually improves results.
"""
import cvUtils
import cv2
import numpy as np
import os.path
import srsGrabcutBgrm as srsgb

if __name__ == "__main__":
    inputFolder = os.path.join('data', 'background')
    outputFolder = os.path.join('data', 'center')

    for filename in cvUtils.imagesInFolder(inputFolder):
        print 'processing ' + filename
        outName = os.path.splitext(os.path.basename(filename))[0] + '.png'
        image = cv2.imread(filename)
        mask = srsgb.srsGrabcutBgRemoval(image, centerPrior = True, centerFactor = 1)
        cv2.imwrite(os.path.join(outputFolder, outName), mask * 255)

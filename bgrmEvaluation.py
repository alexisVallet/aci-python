import sklearn.metrics as skm
import numpy as np
import cv2
import os
import os.path
import cvUtils
import skimage.io as skio

def areaUnderROCCurve(groundTruth, bgrm):
    """ Determines the area under the ROC curve for a background removal mask against
        a ground truth background removal mask.
    """
    return skm.roc_auc_score(groundTruth.flatten('C'), bgrm.flatten('C'))

if __name__ == "__main__":
    groundTruthFolder = os.path.join('data', 'manual_grabcut_bgrm')
    bgrmFolder = os.path.join('data', 'srs_grabcut_bgrm')
    averageAUC = 0
    groundImages = cvUtils.imagesInFolder(groundTruthFolder)
    
    for filename in groundImages:
        groundTruth = skio.imread(filename)
        baseName = os.path.basename(filename)
        bgrm = skio.imread(os.path.join(bgrmFolder, baseName))
        gtMask = cv2.split(groundTruth)[3] / 255
        bgrmMask = cv2.split(bgrm)[3] / 255
        auc = areaUnderROCCurve(gtMask, bgrmMask)
        averageAUC += auc
        print baseName + ': ' + repr(auc)
    averageAUC /= len(groundImages)
    print 'Average AUC: ' + repr(averageAUC)

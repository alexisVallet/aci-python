import os
import os.path
import cvUtils
import cv2
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import sys

# Compute and write background removal metrics to files
if __name__ == "__main__":
    if sys.argv < 3:
        raise ValueError("Please input a ground truth folder and another folder.")
    groundTruthFolder = sys.argv[1]
    comparisonFolders = sys.argv[2:]
    gtFilenames = cvUtils.imagesInFolder(groundTruthFolder)
    nbImages = len(gtFilenames)
    
    for comparisonFolder in comparisonFolders:
        for gtFilename in gtFilenames:
            print 'processing ' + gtFilename
            gtImage = cv2.imread(gtFilename, -1)
            stem = os.path.splitext(os.path.basename(gtFilename))[0]
            comparedImage = cv2.imread(os.path.join(comparisonFolder, stem + '.png'), 
                                       -1)
            confusion = metrics.confusion_matrix(gtImage.flatten('C')/255, 
                                                 comparedImage.flatten('C')/255)
            jsonFile = open(os.path.join(comparisonFolder, stem + '_confusion.json'), 
                            'w')
            json.dump(confusion.tolist(), jsonFile)
            jsonFile.close()

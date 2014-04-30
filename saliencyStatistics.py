import os
import os.path
import cvUtils
import cv2
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import sys

def computeScores(groundTruth, mask):
    """ Returns, in order, the area under the ROC curve, precision tp / (tp + fp) and
        recall tp / (tp + fn).
    """
    trueLabels = groundTruth.flatten('C')/255
    scoreLabels = mask.flatten('C')/255

    return (metrics.roc_auc_score(trueLabels, scoreLabels),
            metrics.precision_score(trueLabels, scoreLabels),
            metrics.recall_score(trueLabels, scoreLabels))

# Compute and write metrics to files
if __name__ == "__main__":
    if sys.argv < 3:
        raise ValueError("Please input a ground truth folder and another folder.")
    groundTruthFolder = sys.argv[1]
    comparisonFolders = sys.argv[2:]
    gtFilenames = cvUtils.imagesInFolder(groundTruthFolder)
    nbImages = len(gtFilenames)
    
    for comparisonFolder in comparisonFolders:
        averages = {'auc':0,'precision':0,'recall':0}
        for gtFilename in gtFilenames:
            print 'processing ' + gtFilename
            gtImage = cv2.imread(gtFilename, -1)
            stem = os.path.splitext(os.path.basename(gtFilename))[0]
            comparedImage = cv2.imread(os.path.join(comparisonFolder, stem + '.png'), -1)
            auc, precision, recall = computeScores(gtImage, comparedImage)
            jsonFile = open(os.path.join(comparisonFolder, stem + '.json'), 'w')
            json.dump({'auc': auc, 'precision': precision, 'recall': recall}, jsonFile)
            jsonFile.close()
            averages['auc'] += auc
            averages['precision'] += precision
            averages['recall'] += recall    
        for scoreName in averages:
            averages[scoreName] /= nbImages
        averagesFile = open(os.path.join(comparisonFolder, 'averages.json'), 'w')
        json.dump(averages, averagesFile)
        averagesFile.close()

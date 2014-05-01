import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

# Displays a plot in ROC space for background removal methods
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please input a folder for at least one method.")
    methodNames = []
    pltNames = []
    i = int(0)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'Indigo', 'BlueViolet', 'Brown', 
              'CadetBlue', 'DarkSalmon', 'LightSeaGreen', 'DarkGreen', 'DarkKhaki',
              'Khaki', 'Fuchsia']

    for methodFolder in sys.argv[1:]:
        print 'processing ' + methodFolder
        methodName = os.path.basename(os.path.normpath(methodFolder))
        totalFP = 0
        totalTP = 0
        totalTN = 0
        totalP = 0
        totalN = 0
        
        for confusionFilename in [f for f in os.listdir(methodFolder) 
                                  if f.endswith('_confusion.json')]:
            confusionFile = open(os.path.join(methodFolder, confusionFilename))
            confusion = json.load(confusionFile)
            confusionFile.close()
            # The confusion matrix has the following shape:
            #     TN  FP
            #     FN  TP
            # And it follows that N = TN + FP and P = FN + TP
            totalFP += confusion[0][1]
            totalTP += confusion[1][1]
            totalTN += confusion[0][0]
            totalP += confusion[1][0] + confusion[1][1]
            totalN += confusion[0][0] + confusion[0][1]
        totalFPR = float(totalFP) / float(totalN)
        totalTPR = float(totalTP) / float(totalP)
        accuracy = float(totalTP + totalTN) / float(totalP + totalN)
        marker = 'x' if 'center' in methodName else '.'
        pltName = plt.scatter([totalFPR], [totalTPR], marker=marker, 
                              color=colors[int(i)], s=40)
        methodNames.append(methodName)
        pltNames.append(pltName)
        print repr(accuracy)
        i += int(1)
    plt.plot([0,1],[0,1], 'k--')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.axis([0.1,0.4,0.6,0.9])
    plt.grid(True)
    plt.legend(pltNames, methodNames, scatterpoints=1, loc='lower left', ncol=4, fontsize=8)
    plt.show()

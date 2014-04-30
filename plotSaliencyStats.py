import numpy as np
import matplotlib.pyplot as plt
import os.path
import json

def loadFolderJson(folder):
    jsonFile = open(os.path.join('data', folder, 'averages.json'))
    jsonData = json.load(jsonFile)
    jsonFile.close()
    return jsonData

if __name__ == "__main__":
    bgr1center = loadFolderJson('BGR_1_center')
    bgr2center = loadFolderJson('BGR_2_center')
    bgr3center = loadFolderJson('BGR_3_center')
    lab1center = loadFolderJson('LAB_1_center')
    lab2center = loadFolderJson('LAB_2_center')
    lab3center = loadFolderJson('LAB_3_center')
    center = loadFolderJson('center')
    bgr1 = loadFolderJson('BGR_1')
    bgr2 = loadFolderJson('BGR_2')
    bgr3 = loadFolderJson('BGR_3')
    lab1 = loadFolderJson('LAB_1')
    lab2 = loadFolderJson('LAB_2')
    lab3 = loadFolderJson('LAB_3')

    for scoreName in ['auc', 'precision', 'recall']:
        plt.plot([1,2,3], 
                 [bgr1center[scoreName], bgr2center[scoreName], bgr3center[scoreName]],
                 'b',
                 label='BGR with center')
        plt.plot([1,2,3], 
                 [lab1center[scoreName], lab2center[scoreName], lab3center[scoreName]],
                 'r',
                 label='LAB with center')
        plt.plot([1,2,3], [center[scoreName]] * 3, 'g', label='Center alone')
        plt.plot([1,2,3], [bgr1[scoreName], bgr2[scoreName], bgr3[scoreName]], 'b--',
                 label='BGR alone')
        plt.plot([1,2,3], [lab1[scoreName], lab2[scoreName], lab3[scoreName]], 'r--',
                 label='LAB alone')
        plt.legend(('BGR with center', 'LAB with center', 'Center alone', 'BGR alone', 'LAB alone'), loc='lower right')
        plt.ylabel('average ' + scoreName)
        plt.ylim([0.5,1])
        plt.xlabel('Number of channels used.')
        plt.show()

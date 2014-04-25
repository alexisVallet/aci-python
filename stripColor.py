""" Simple script to only keep the alpha channel of .png images, to save space in the
git repo. Handle with care.
"""
import os
import os.path
import sys
import cv2
import cvUtils

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Please input a folder name")
    folder = sys.argv[1]
    yesNo = raw_input("This is going to keep only the alpha channel of all images in " + folder + ", do you want to continue ? [yes/no]")
    if yesNo == "yes":
        for filename in cvUtils.imagesInFolder(folder):
            [b, g, r, a] = cv2.split(cv2.imread(filename, -1))
            cv2.imwrite(filename, a)
        

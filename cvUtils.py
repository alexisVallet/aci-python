import os
import numpy as np
import cv2
import objectSegmentation as objseg
import array
import random

def showScaled(winName, grayscaleImage):
    maxVal = np.amax(grayscaleImage)
    minVal = np.amin(grayscaleImage)
    print winName + ' min: ' + repr(minVal) + ', max: ' + repr(maxVal)
    cv2.imshow(winName, (grayscaleImage - minVal) / (maxVal - minVal))
    cv2.waitKey(0)

class Connectivity:
    four_connectivity = [(1, 0), (0,1)]
    eight_connectivity = four_connectivity + [(1,1), (-1,1)]

def connectedComponents(binaryImage, connectivity = Connectivity.four_connectivity):
    """ Returns the connected components of value 1 in a binary image.
    Args:
        binaryImage (array): arbitrary binary image.
        connectivity (list): connectivity to consider, can be either 
        Connectivity.four_connectivity or Connectivity.eight_connectivity.
    Returns:
        A pair (components, background) where components is a disjoint set forest of
        the connected components of the image, background is an element of the 0-valued
        segment - None if there is no such segment.
    """
    rows, cols = binaryImage.shape[0:2]
    components = objseg.ObjectsSegmentation(binaryImage)
    background = None

    for i in range(0,rows):
        for j in range(0,cols):
            pixelValue = binaryImage[i,j]
            # If it's a background pixel, set it so
            if pixelValue == 0:
                components.setBackground(i, j)
            # Otherwise, fuse it with its neighbors who are also non-background
            else:
                for (offI,offJ) in connectivity:
                    nI, nJ = (i + offI, j + offJ)
                    if 0 <= nI < rows and 0 <= nJ < cols:
                        if binaryImage[nI, nJ] != 0:
                            components.union(i, j, nI, nJ)
    return (components, background)

def segmentationImage(image, segmentation):
    random.seed()
    rootToColor = {}
    rows, cols = image.shape
    segmentationImage = np.empty([rows, cols, 3])

    for i in range(0,rows):
        for j in range(0,cols):
            segmentRoot = segmentation.find(toRowMajor(cols,i,j))
            if not segmentRoot in rootToColor:
                rootToColor[segmentRoot] = (random.randint(0, 255),
                                            random.randint(0, 255),
                                            random.randint(0, 255))
            segmentationImage[i,j] = rootToColor[segmentRoot]
    return segmentationImage

def boundingBoxes(binaryImage, connectivity = Connectivity.four_connectivity):
    """ From a binary image where 1 represents an object, 0 the background, returns
        bounding boxes for objects corresponding to connected components.
    Args:
        binaryImage (array): binary image where 1 represents an object, 0 the 
            background.
        connectivity (list): connectivity to take into account for objects, must be 
            either Connectivity.four_connectivity or Connectivity.eight_connectivity .
    Returns:
        A list of tuples of the shape (i,j,i',j') where (i,j) is the position of the
        top-left corner of the box, and (i',j') is the position of the bottom-right
        corner in (row,col) order.
    """
    components, background = connectedComponents(binaryImage, connectivity)
    boundingBoxes = {}
    rows, cols = binaryImage.shape

    for i in range(0, rows):
        for j in range(0, cols):
            root = components.find(i,j)
            # ignore the background
            if root == background:
                pass
            elif not root in boundingBoxes:
                boundingBoxes[root] = (i,j,i,j)
            else:
                tli, tlj, bri, brj = boundingBoxes[root]
                boundingBoxes[root] = (min(tli, i), min(tlj, j), 
                                       max(bri, i), max(brj, j))

    return boundingBoxes.values()

def generateMask(segmentation, segmentRoot, segVal = 1, bgVal = 0):
    """ Generates a mask image for a single segment.
    Args:
        image (array): segmented image.
        segmentation (ObjectSegmentation): segmentation of the image.
        segmentRoot (tuple): root pixel of the segment to create a mask for.
        segVal: value for the segment pixels in the mask.
        bgVal: value for the background in the mask
    Returns:
        A mask image for the segment.
    """
    rows, cols = segmentation.image.shape[0:2]
    mask = np.ones([rows, cols], dtype=np.uint8)
    mask *= bgVal
    
    for i in range(0,rows):
        for j in range(0,cols):
            root = segmentation.find(i,j)
            if root == segmentRoot:
                mask[i,j] = segVal
    
    return mask

def applyMask(image, mask):
    """ Applies a mask to a color image.
    """
    rows, cols = image.shape[0:2]
    maskedImage = np.empty_like(image)
    
    for i in range(0,rows):
        for j in range(0,cols):
            maskedImage[i,j] = image[i,j] * mask[i,j]
    return maskedImage

def maskAsAlpha(image, mask):
    """ Adds an alpha channel to an 8-bit image using information from a mask.
    """
    alayer = (mask * 255).astype(np.uint8)
    blayer, glayer, rlayer = cv2.split(image)
    return cv2.merge([blayer, glayer, rlayer, alayer])

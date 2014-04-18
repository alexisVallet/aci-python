import disjointSetForest as dsj
import cv2
import numpy as np

def toRowMajor(cols, i, j):
    return i * cols + j

def fromRowMajor(cols, idx):
    return (idx / cols, idx % cols)

class ObjectsSegmentation:
    """ Disjoint set forest, with the additional semantic element of an image to segment
        into background and objects (foreground).
    """
    def __init__(self, image):
        rows, cols = image.shape[0:2]
        self.image = image
        self.segmentation = dsj.DisjointSetForest(rows * cols)
        self.background = None
        self.largest = None

    def find(self, i, j):
        """ Finds the root pixel of the segment containing pixel (i,j).
        """
        rows, cols = self.image.shape[0:2]

        return fromRowMajor(cols, 
                            self.segmentation.find(toRowMajor(cols, i, j)))

    def unsafeUnion(self, i, j, k, l):
        """ Fuses the segments containing pixels (i,j) and (k,l) into a single segment.
            Doesn't check if either segment is the background.
        """
        rows, cols = self.image.shape[0:2]
        newRoot = self.segmentation.union(toRowMajor(cols,i,j),
                                          toRowMajor(cols,k,l))
        return fromRowMajor(cols, newRoot)

    def union(self, i, j, k, l):
        """ Fuses the segments containing pixels (i,j) and (k,l) into a single segment.
            Neither segments should be the background.
        """
        rows, cols = self.image.shape[0:2]
        fstRoot = self.find(i,j)
        sndRoot = self.find(k,l)
        if fstRoot == self.background or sndRoot == self.background:
            raise ValueError("Cannot perform union of background pixels!")
        else:
            newRoot = self.segmentation.union(toRowMajor(cols,i,j),
                                              toRowMajor(cols,k,l))
            newRootPixel = fromRowMajor(cols,newRoot)
            # keep track of the largest object
            if self.largest == None:
                self.largest = newRootPixel
            else:
                (li, lj) = self.largest
                largestSize = self.segmentation.compSize[toRowMajor(cols,li,lj)]
                if self.segmentation.compSize[newRoot] > largestSize:
                    self.largest = newRootPixel

    def setBackground(self, i, j):
        """ Marks the (i,j) pixel as a background pixel.
        """
        if self.background == None:
            self.background = (i,j)
        else:
            (k,l) = self.background
            self.background = self.unsafeUnion(k, l, i, j)

    def getLargestObject(self):
        return (0,0) if self.largest == None else self.largest

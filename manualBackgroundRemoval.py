import cv2
import numpy as np
import os
import os.path
import cvUtils

class RectangleDrawing:
    def __init__(self, sourceImage):
        self.upperLeft = None
        self.downRight = None
        self.sourceImage = sourceImage
        self.image = np.array(sourceImage, copy=True)

    def drawRectangle(self, event, x, y):
        # Allow the user to change his selection
        if self.upperLeft != None and self.downRight != None and event == cv2.EVENT_LBUTTONDOWN:
            self.upperLeft = None
            self.downRight = None
        if self.upperLeft == None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.upperLeft = (x,y)
                print 'upperLeft = ' + repr(self.upperLeft)
        elif event == cv2.EVENT_MOUSEMOVE and self.downRight == None:
            self.image = np.array(self.sourceImage, copy=True)
            cv2.rectangle(self.image, self.upperLeft, (x,y), (0,0,255),1)
        elif self.downRight == None:
            if event == cv2.EVENT_LBUTTONUP:
                self.downRight = (x,y)
                print 'downRight = ' + repr(self.downRight)

class SwitchBGFGSegments:
    def __init__(self, image, gcMask, bgModel, fgModel):
        self.sourceImage = image
        self.gcMask = gcMask
        self.bgModel = bgModel
        self.fgModel = fgModel
        self.image = np.empty_like(image)
        self.draw()

    def switchLabel(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            # modify mask accordingly
            if self.gcMask[y,x] == cv2.GC_BGD or cv2.GC_PR_BGD:
                self.gcMask[y,x] = cv2.GC_FGD
            else:
                self.gcMask[y,x] = cv2.GC_BGD
            # run grabcut again
            cv2.grabCut(self.sourceImage, self.gcMask, None, self.bgModel, self.fgModel,
                        1, cv2.GC_INIT_WITH_MASK)
            # draw the results
            self.draw()

    def draw(self):
        rows, cols = self.sourceImage.shape[0:2]
        
        for i in range(0,rows):
            for j in range(0,cols):
                if self.gcMask[i,j] == cv2.GC_BGD:
                    self.image[i,j] = self.sourceImage[i,j] * 0.25
                elif self.gcMask[i,j] == cv2.GC_PR_BGD:
                    self.image[i,j] = self.sourceImage[i,j] * 0.5
                elif self.gcMask[i,j] == cv2.GC_PR_FGD:
                    self.image[i,j] = self.sourceImage[i,j]
                else:
                    self.image[i,j] = (0,0,255)

def manualGrabcutRemoval(bgrImage):
    """ Runs the grabcut algorithm to remove the background of an image, prompting the
        user for input. First asks for a bounding box, then segments to add to the
        foreground (left click) or background (right click).
    """
    # Ask the user for a bounding box
    cv2.namedWindow('image')
    rectangleDrawing = RectangleDrawing(bgrImage)
    cv2.setMouseCallback('image', 
                         lambda e,x,y,f,p: rectangleDrawing.drawRectangle(e, x, y))
    key = ord('a')
    while (key != ord('n') 
           or rectangleDrawing.upperLeft == None 
           or rectangleDrawing.downRight == None):
        cv2.imshow('image', rectangleDrawing.image)
        key = cv2.waitKey(int(1000/60))

    # Apply grabcut using this bounding box
    fgModel, bgModel = [None] * 2
    mask = np.zeros(bgrImage.shape[0:2], dtype=np.uint8)
    (ulx, uly, drx, dry) = rectangleDrawing.upperLeft + rectangleDrawing.downRight
    cv2.grabCut(bgrImage, mask,
                (ulx, uly, drx - ulx, dry - uly),
                bgModel, fgModel, 1, cv2.GC_INIT_WITH_RECT)

    # Iteratively apply grabcut
    regularMask = np.frompyfunc(lambda i: 1 if i == cv2.GC_PR_FGD else 0, 1, 1)(mask).astype(np.uint8)
    maskedImage = cvUtils.applyMask(bgrImage, regularMask)
    bgSwitcher = SwitchBGFGSegments(bgrImage, mask, bgModel, fgModel)
    cv2.setMouseCallback('image', lambda e,x,y,f,p: bgSwitcher.switchLabel(e,x,y))

    key = ord('a')
    while key != ord('n'):
        cv2.imshow('image', bgSwitcher.image)
        cv2.waitKey(int(1000/60))
    
        

if __name__ == "__main__":
    image = cv2.imread(os.path.join('data', 'background', 'asuka_langley_3.jpg'))
    manualGrabcutRemoval(image)

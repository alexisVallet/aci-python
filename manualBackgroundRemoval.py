import os
import os.path
from subprocess import call

inputFolder = 'data/background'
outputFolder = 'data/manual_grabcut_bgrm'
imageFilenames = [os.path.join(inputFolder, f) for f in sorted(os.listdir(inputFolder), key=str.lower)
                  if os.path.isfile(os.path.join(inputFolder, f)) 
                  and f.lower().endswith(('.png', '.jpg', '.gif'))
                  and not (os.path.splitext(f)[0] + '.png' in os.listdir(outputFolder))]

for filename in imageFilenames:
    call(["./GrabCutExe", filename, outputFolder])

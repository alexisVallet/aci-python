import os
import os.path
from subprocess import call

inputFolder = 'data/background'
outputFolder = 'data/manual_grabcut_bgrm'
imageFilenames = [os.path.join(inputFolder, f) for f in sorted(os.listdir(inputFolder), key=str.lower)
                  if os.path.isfile(os.path.join(inputFolder, f)) 
                  and f.lower().endswith(('.png', '.jpg', '.gif'))]

for filename in imageFilenames[10:]:
    call(["./GrabCutExe", filename, outputFolder])

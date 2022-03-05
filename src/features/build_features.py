#convert the jpg files to npy
#argv[1]=jpg directory
#argv[2]=npy directory
import os
import PIL
from PIL import Image
import numpy as np
from numpy import save
import  sys
#path to project
path='/content/drive/My Drive/Datasets/caliche/severstal-steel-defect-detection'
#jpgPath is the path to a directory containing all jpg files to be converted
jpgPath=path+'/'+sys.argv[1]+'/'
#npyPath is the path to an empty directory to store all npy files.
npyPath=path+'/'+sys.argv[2]+'/'
dir=os.listdir(jpgPath)
dir.sort()
for item in dir:
  if os.path.isfile(jpgPath+item):
    img=Image.open(jpgPath+item).convert("RGB")
    img=np.array(img)
    size=len(item)
    np.save(npyPath+item[:size-4]+'.npy',img)
    del(img)


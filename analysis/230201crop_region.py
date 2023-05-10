import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

images_dir = "/media/sf_research/cass/code/data/"

left = 100
top = 260
right = 950
bottom = 900
angle = 0

pixels = (right-left)*(top-bottom)
height = top-bottom 
width = right-left 

#2d image array (height, width)
#each image is stored row after row in a 1d array
images_path = []
for file in os.listdir(images_dir):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir,file))


im_list = []
c=0
for image in images_path:
    im = Image.open(image)
    im = im.rotate(angle)
    im_cropped = im.crop((left,top,right,bottom))
    im_cropped.save(images_dir+str(c)+".tiff")
    
    c = c+1


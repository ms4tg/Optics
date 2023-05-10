import os
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

angle = 0

images_dir = "/media/sf_research/cass/code/output/230314ic_upper_all/"
images_path = []
for file in os.listdir(images_dir):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir,file))


s={}
for image in images_path:
    #print (image)
    n_pc = re.findall(r'(\d+)\.tif',image)
    #print(n_pc)
    im = Image.open(image)
    im = im.rotate(angle)
    left = 64
    top = 40
    right = 79
    bottom = 55
    angle = 0
    im_cropped = im.crop((left,top,right,bottom))
    imarray = np.array(im_cropped)
    s1 = np.sum(imarray)  

    left = 161
    top = 36
    right = 176
    bottom = 51
    im_cropped = im.crop((left,top,right,bottom))
    imarray = np.array(im_cropped)
    s2 = np.sum(imarray)  
    
    left = 257
    top = 34
    right = 272
    bottom = 49
    im_cropped = im.crop((left,top,right,bottom))
    imarray = np.array(im_cropped)
    s3 = np.sum(imarray)  

    s[n_pc[0]] = abs(s1+s3-2*s2)

sorted_s = dict(sorted(s.items(),key=lambda x:x[1]))
print(sorted_s)

with open(images_dir+'ic_upper.csv', 'w') as f:
    writer = csv.writer(f)
    for row in sorted_s.items():
        writer.writerow(row)

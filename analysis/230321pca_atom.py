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

images_dir = "/media/sf_research/cass/code/output/230314ic_lower_all/"
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
    top = 46
    right = 79
    bottom = 61
    im_cropped = im.crop((left,top,right,bottom))
    imarray = np.array(im_cropped)
    s1 = np.sum(imarray)  
    

    left = 163
    top = 46
    right = 178
    bottom = 61
    im_cropped = im.crop((left,top,right,bottom))
    imarray = np.array(im_cropped)
    s2 = np.sum(imarray)  
    
    left = 256
    top = 40
    right = 271
    bottom = 55
    im_cropped = im.crop((left,top,right,bottom))
    imarray = np.array(im_cropped)
    s3 = np.sum(imarray)  

    s[n_pc[0]] = abs(s1+s3-2*s2)

sorted_s = dict(sorted(s.items(),key=lambda x:x[1]))
#print(sorted_s)

with open(images_dir+'ic_lower.csv', 'w') as f:
    writer = csv.writer(f)
    for row in sorted_s.items():
        writer.writerow(row)

import os
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

images_dir = "/media/sf_research/cass/code/output/230314ic_lower_all/"
#images_dir1 = "/media/sf_BIGTOP_data/1112/"
images_dir1 = "/Users/menglin/Library/CloudStorage/Dropbox/BIGTOP data/1112/"
#images_dir2 = "/media/sf_BIGTOP_data/1113/"
images_dir2 = "/Users/menglin/Library/CloudStorage/Dropbox/BIGTOP data/1113/"
output_dir = "/media/sf_research/cass/code/output/230405ic_recon_notscale/"

blacklist = [0,3,5,14,16,19,23,36,42] 

angle = 45
left = 220
top = 767
right = 530
bottom = 917

pixels = (right-left)*(top-bottom)
height = top-bottom 
width = right-left 

images_path = []
for file in os.listdir(images_dir1):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir1,file))

for file in os.listdir(images_dir2):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir2,file))

im_list_below = []
for image in images_path:
    im = Image.open(image)
    im = im.rotate(angle)
    im_cropped = im.crop((left,top,right,bottom))
    imarray = np.array(im_cropped)
    imarray = imarray.flatten()
    im_list_below.append(imarray)    

images_path = []
for file in os.listdir(images_dir):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir,file))

weight = np.loadtxt(images_dir+"lower_weight.txt")
signal_weight =pd.DataFrame(weight)

s={}
for image in images_path:
    #print (image)
    n_pc = re.findall(r'(\d+)\.tif',image)
    #print(n_pc)
    im = Image.open(image)
    imarray = np.array(im)
    imarray = imarray.flatten()
    s[int(n_pc[0])] = imarray

index = list(s.keys())
index.sort()
sorted_ic = {i: s[i] for i in index}
ICs=pd.DataFrame.from_dict(sorted_ic,orient='index').T
ICs = ICs.drop(columns=blacklist, axis = 1)

print (ICs.T.head)
signal_weight= signal_weight.drop(columns=blacklist, axis = 1)

print(pd.DataFrame(signal_weight).head)

data1 = (signal_weight.dot(ICs.T)).to_numpy()
data2 = im_list_below

plt.figure()
for i in range(len(data1)):
    plt.clf()
    fig, axis = plt.subplots(1, 2,figsize=(20,10))
    axis[0].imshow(data2[i].reshape(height, width), cmap="gray")
    axis[1].imshow(data1[i].reshape(height, width), cmap="gray")
    axis[0].set_title("Original Image " + str(i), fontsize=12)
    axis[1].set_title("Reconstructed Image" + str(i), fontsize=12)
    plt.savefig(output_dir+"lower_image"+str(i)+".jpg")
    #data1[i] = (data1[i]-np.min(data1[i]))/(np.max(data1[i])-np.min(data1[i]))
    #print(data1[i])
    #data2[i] = (data2[i]-np.min(data2[i]))/(np.max(data2[i])-np.min(data2[i]))
    #ic_lower = Image.fromarray(data1[i].reshape(height,width),"F")
    #ic_lower = Image.fromarray(np.uint8((data1[i].reshape(height,width))*255),"L")
    #ic_lower.save(output_dir+"lower_image"+str(i)+".tif")
    #ic_upper.save(output_dir2+"uppwe_pc"+str(i)+".tif")

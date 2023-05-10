import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

images_dir1 = "/media/sf_BIGTOP_data/1112/"
images_dir2 = "/media/sf_BIGTOP_data/1113/"
output_dir1 = "/media/sf_research/cass/code/output/230321pc_lower_all/"
output_dir2 = "/media/sf_research/cass/code/output/230321pc_upper_all/"

angle = 45
n_pc = 50

#2d image array (height, width)
#each image is stored row after row in a 1d array
images_path = []
for file in os.listdir(images_dir1):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir1,file))

for file in os.listdir(images_dir2):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir2,file))

left = 220
top = 520
right = 530
bottom = 670

im_list_top = []
for image in images_path:
    im = Image.open(image)
    im = im.rotate(angle)
    im_cropped = im.crop((left,top,right,bottom))
    #im_cropped.save(images_dir+"0.tiff")  #check the cropped area, save in tiff
    imarray = np.array(im_cropped)
    #print(imarray.shape)
    imarray = imarray.flatten()
    #print(imarray.shape)
    im_list_top.append(imarray)    

left = 220
top = 767
right = 530
bottom = 917

im_list_below = []
for image in images_path:
    im = Image.open(image)
    im = im.rotate(angle)
    im_cropped = im.crop((left,top,right,bottom))
    #im_cropped.save(images_dir+"0.tiff")  #check the cropped area, save in tiff
    imarray = np.array(im_cropped)
    #print(imarray.shape)
    imarray = imarray.flatten()
    #print(imarray.shape)
    im_list_below.append(imarray)    


X_top = pd.DataFrame(im_list_top)
#scaler = StandardScaler()
#scaler.fit(X)
#X_top = scaler.transform(X)

pca = PCA(n_components =None, svd_solver='randomized', random_state = 42)
pca.fit(X_top)
X_pca = pca.transform(X_top)

X_below = pd.DataFrame(im_list_below)
#scaler = StandardScaler()
#scaler.fit(X)
#X_below = scaler.transform(X)

pca_s = PCA(n_components =None, svd_solver='randomized', random_state = 42)
pca_s.fit(X_below)
X_pca_s = pca_s.transform(X_below)

df_pca_loadings = pd.DataFrame(pca.get_covariance())
print(df_pca_loadings.head)


pixels = (right-left)*(top-bottom)
height = top-bottom 
width = right-left 

data1 = pca.components_
data2 = pca_s.components_


for i in range(len(data1)):
    data1[i] = (data1[i]-np.min(data1[i]))/(np.max(data1[i])-np.min(data1[i]))
    data2[i] = (data2[i]-np.min(data2[i]))/(np.max(data2[i])-np.min(data2[i]))
    ic_lower = Image.fromarray(data1[i].reshape(height,width),"F")
    ic_upper = Image.fromarray(data2[i].reshape(height,width),"F")
    ic_lower.save(output_dir1+"lower_pc"+str(i)+".tif")
    ic_upper.save(output_dir2+"upper_pc"+str(i)+".tif")
#plt.imshow(data2.reshape(height, width), cmap="binary",alpha = 1,interpolation='none')
#plt.imshow(data1.reshape(height, width), cmap="jet",alpha = 0.5,interpolation='none')
#plt.savefig(output_dir+"pc9_im.jpg")

#plt.clf()

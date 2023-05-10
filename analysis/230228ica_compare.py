import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

#images_dir = "/media/sf_research/cass/code/data/"
images_dir1 = "/media/sf_BIGTOP_data/1112/"
images_dir2 = "/media/sf_BIGTOP_data/1113/"
output_dir= "/media/sf_research/cass/code/output/230321pc_lower_all/"

left = 220
top = 765
right = 530
bottom = 915
angle = 45

pixels = (right-left)*(top-bottom)
height = top-bottom 
width = right-left 

#2d image array (height, width)
#each image is stored row after row in a 1d array
images_path = []
for file in os.listdir(images_dir1):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir1,file))

for file in os.listdir(images_dir2):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir2,file))

im_list = []
for image in images_path:
    im = Image.open(image)
    im = im.rotate(angle)
    im_cropped = im.crop((left,top,right,bottom))
    #im_cropped.save(images_dir+"0.tiff")  #check the cropped area, save in tiff
    imarray = np.array(im_cropped)
    #print(imarray.shape)
    imarray = imarray.flatten()
    #print(imarray.shape)
    im_list.append(imarray)    
#print(im_list)

#in dataframe, 
#all the pixels for one image is stored in one row --observables
#pixels at the same position is stored in one column --factors
X = pd.DataFrame(im_list)
#print(X.head)
#print(X.shape)

#data standardization 
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#print(X_scaled.head)

ica = FastICA(n_components = 50, whiten='unit-variance')
#ica.fit(X_scaled)
X_ica = ica.fit_transform(X_scaled)

print("X_ica\n")
print(pd.DataFrame(X_ica).head)

#X_new = scaler.inverse_transform(components)
#print(X_new.head)

X_recon = ica.inverse_transform(X_ica)
#X_new = scaler.inverse_transform(X_recon)

for i in range(50):
    plt.clf()
    fig, axis = plt.subplots(1, 1,figsize=(10,10))
    axis.imshow(ica.components_[i].reshape(height, width), cmap="binary")
    axis.set_title("Independent Component " + str(i), fontsize=12)
    plt.savefig(output_dir+"lower_IC_"+str(i)+".jpg")

    plt.clf()
    fig, axis = plt.subplots(1, 1,figsize=(10,10))
    axis.imshow(X_recon[i].reshape(height, width), cmap="binary")
    axis.set_title("Reconctructed Image " + str(i), fontsize=12)
    plt.savefig(output_dir+"Recon_"+str(i)+".jpg")
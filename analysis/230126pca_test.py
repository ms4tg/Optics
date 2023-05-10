import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#images_dir = "/media/sf_research/cass/code/data/"
images_dir1 = "/media/sf_BIGTOP_data/1112/"
images_dir2 = "/media/sf_BIGTOP_data/1113/"

left = 220
top = 520
right = 530
bottom = 670
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
for image in images_path[:20]:
    im = Image.open(image)
    im = im.rotate(45)
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
print(X.head)
print(X.shape)

#data standardization 
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

pca_95 = PCA(n_components = 5, random_state = 42)
pca_95.fit(X_scaled)
X_pca_95 = pca_95.transform(X_scaled)

plt.plot(np.cumsum(pca_95.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Principle Componentes Explaining 95% Variance")
plt.savefig("comulative_var.jpg")
plt.close()

components = pca_95.components_
print(components.shape)
print(components)

#X_new = scaler.inverse_transform(components)
#print(X_new.head)

X_recon = pca_95.inverse_transform(X_pca_95)
X_new = scaler.inverse_transform(X_recon)
print(X_new.shape)

for i in range(50):
    im_recon_1d = np.asarray(X_new[i])
    im_recon_2d = im_recon_1d.reshape(height, width)
    plt.figure()
    fig, axis = plt.subplots(1, 1,figsize=(10,10))
    axis.imshow(im_recon_2d, cmap="binary")
    axis.set_title("Reconstructed Image " + str(i), fontsize=12)
    plt.savefig("Recon_"+str(i)+".jpg")
    plt.close()
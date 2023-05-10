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
output_dir = "/media/sf_research/cass/code/output/230221weight/"

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
print(X.head)
#print(X.shape)

#data standardization 
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#print(X_scaled.head)

pca = PCA(n_components = None)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print("X_pca\n")
print(pd.DataFrame(X_pca).head)


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Variance Explained by Principle Componentes")
plt.savefig(output_dir+"comulative_var.jpg")

plt.clf()
plt.bar(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('percentage variance explained')
plt.title("Variance Explained by Principle Componentes")
plt.savefig(output_dir+"individual_var.jpg")

df_pca_loadings = pd.DataFrame(pca.get_covariance())
print(df_pca_loadings.head)

df_pca_loadings = pd.DataFrame(pca.components_)
print(df_pca_loadings.head)

#X_new = scaler.inverse_transform(components)
#print(X_new.head)

X_recon = pca.inverse_transform(X_pca)
#X_new = scaler.inverse_transform(X_recon)

for i in range(50):
    plt.clf()
    fig, axis = plt.subplots(1, 1,figsize=(10,10))
    axis.imshow(df_pca_loadings.loc[i].values.reshape(height, width), cmap="binary")
    axis.set_title("Principle Component " + str(i), fontsize=12)
    plt.savefig(output_dir+"lower_PC_"+str(i)+".jpg")

quit()  
plt.clf()
fig, axis = plt.subplots(1, 1,figsize=(10,10))
axis.imshow(X_recon[i].reshape(height, width), cmap="binary")
axis.set_title("Reconctructed Image " + str(i), fontsize=12)
plt.savefig(output_dir+"Recon_"+str(i)+".jpg")
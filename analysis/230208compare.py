import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#images_dir = "/media/sf_research/cass/code/data/"
images_dir1 = "/media/sf_BIGTOP_data/1112/"
images_dir2 = "/media/sf_BIGTOP_data/1113/"
output_dir = "/media/sf_research/cass/code/output/230214rotation2/"

left = 220
top = 765
right = 530
bottom = 915
angle = 45
n_pc = 10

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

pca = PCA(n_components =n_pc, svd_solver='randomized', random_state = 42)
pca.fit(X)
X_pca = pca.transform(X)

pca_s = PCA(n_components =50, svd_solver='randomized', random_state = 42)
pca_s.fit(X_scaled)
X_pca_s = pca_s.transform(X_scaled)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Variance Explained by Principle Componentes")
plt.savefig(output_dir+"0comulative_var.jpg")

plt.clf()
plt.bar(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('percentage variance explained')
plt.title("Variance Explained by Principle Componentes")
plt.savefig(output_dir+"0individual_var.jpg")

plt.figure()
plt.plot(np.cumsum(pca_s.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Variance Explained by Principle Componentes")
plt.savefig(output_dir+"0comulative_var_scaled.jpg")

plt.clf()
plt.bar(range(0,len(pca_s.explained_variance_ratio_)), pca_s.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('percentage variance explained')
plt.title("Variance Explained by Principle Componentes")
plt.savefig(output_dir+"0individual_var_scaled.jpg")

#X_new = scaler.inverse_transform(components)
#print(X_new.head)

X_recon = pca.inverse_transform(X_pca)
#X_new = scaler.inverse_transform(X_recon)
data1 = pca.components_
data2 = pca_s.components_

n_image = 50
if (n_pc<50):
    n_image = n_pc

for i in range(int(n_image)):
    plt.clf()
    fig, axis = plt.subplots(1, 2,figsize=(20,10))
    axis[0].imshow(data1[i].reshape(height, width), cmap="binary")
    axis[1].imshow(data2[i].reshape(height, width), cmap="binary")
    axis[0].set_title("Principle Component " + str(i)+" out of "+str(n_pc), fontsize=12)
    axis[1].set_title("Scaled Principle Component " + str(i)+"out of "+str(n_pc), fontsize=12)
    plt.savefig(output_dir+"PC_compare"+str(i)+".jpg")



quit()

def recon_at_k(k,data):
    pca = PCA(n_components =k, svd_solver='randomized', random_state = 42)
    image_recon = pca.inverse_transform(pca.fit_transform(data))
    return image_recon

ks = [5, 10, 15, 20, 25, 50]

for i in range(5):
    plt.clf()
    index = random.randrange(len(X_recon))
    fig, axis = plt.subplots(2, 6,figsize=(40,10), sharex = True, sharey = True)
    for j in range(len(ks)):
        data1 = recon_at_k(ks[j], X)
        axis[0,j].imshow(data1[index].reshape(height, width), cmap="binary")
        axis[0,j].set_title("Components: "+str(ks[j]))

        data2 = recon_at_k(ks[j], X_scaled)
        axis[1,j].imshow(data2[index].reshape(height, width), cmap="binary")
        axis[1,j].set_title("Scaled Components: "+str(ks[j]))

    #axis.set_title("Reconctructed Image " + str(i), fontsize=12)
    plt.savefig(output_dir+"Recon_compare"+str(i)+".jpg")

data1 = pca.components_
data2 = pca_s.components_
for i in range(50):
    plt.clf()
    fig, axis = plt.subplots(1, 2,figsize=(20,10))
    axis[0].imshow(data1[i].reshape(height, width), cmap="binary")
    axis[1].imshow(data2[i].reshape(height, width), cmap="binary")
    axis[0].set_title("Principle Component " + str(i), fontsize=12)
    axis[1].set_title("Scaled Principle Component " + str(i), fontsize=12)
    plt.savefig(output_dir+"PC_compare"+str(i)+".jpg")


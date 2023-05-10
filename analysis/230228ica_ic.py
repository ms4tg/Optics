import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

#images_dir = "/media/sf_research/cass/code/data/"
images_dir1 = "/media/sf_BIGTOP_data/1112/"
images_dir2 = "/media/sf_BIGTOP_data/1113/"
output_dir1 = "/media/sf_research/cass/code/output/230314ic_lower_all/"
output_dir2 = "/media/sf_research/cass/code/output/230314ic_upper_all/"

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

angle = 45
n_pc = 50


im_list_top = []
im_list_below = []
for image in images_path:
    im = Image.open(image)
    im = im.rotate(angle)
    left = 220
    top = 520
    right = 530
    bottom = 670
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
    im_cropped = im.crop((left,top,right,bottom))
    #im_cropped.save(images_dir+"0.tiff")  #check the cropped area, save in tiff
    imarray = np.array(im_cropped)
    #print(imarray.shape)
    imarray = imarray.flatten()
    #print(imarray.shape)
    im_list_below.append(imarray)

    

#in dataframe, 
#all the pixels for one image is stored in one row --observables
#pixels at the same position is stored in one column --factors
X_scaled = pd.DataFrame(im_list_top)
#print(X.head)
#print(X.shape)

#data standardization 
#scaler = StandardScaler()
#scaler.fit(X)
#X_scaled = scaler.transform(X)

#print(X_scaled.head)

ica = FastICA(n_components = 50, whiten='unit-variance', max_iter = 200)
#ica.fit(X_scaled)
X_ica = ica.fit_transform(X_scaled)

X_scaled_s = pd.DataFrame(im_list_below)
#scaler = StandardScaler()
#scaler.fit(X)
#X_scaled_s = scaler.transform(X)
ica_s = FastICA(n_components = 50, whiten='unit-variance', max_iter = 200)
X_ica_s = ica_s.fit_transform(X_scaled_s)

np.savetxt(output_dir2+"upper_weight.txt",X_ica,fmt='%f')
np.savetxt(output_dir1+"lower_weight.txt",X_ica_s,fmt='%f')
np.savetxt(output_dir2+"upper_mean.txt",ica.mean_,fmt='%f')
np.savetxt(output_dir1+"lower_mean.txt",ica_s.mean_,fmt='%f')


print("X_ica\n")
print(pd.DataFrame(X_ica).head)

#X_new = scaler.inverse_transform(components)
#print(X_new.head)

X_recon = ica.inverse_transform(X_ica)
#X_new = scaler.inverse_transform(X_recon)

data1 = ica.components_
data2 = ica_s.components_
data3 = pd.DataFrame(X_ica).T.iloc[5]
data4 = pd.DataFrame(X_ica_s).T.iloc[9]


for i in range(len(data1)):
    data1[i] = (data1[i]-np.min(data1[i]))/(np.max(data1[i])-np.min(data1[i]))
    data2[i] = (data2[i]-np.min(data2[i]))/(np.max(data2[i])-np.min(data2[i]))
    ic_lower = Image.fromarray(data2[i].reshape(height,width),"F")
    ic_upper = Image.fromarray(data1[i].reshape(height,width),"F")
    ic_lower.save(output_dir1+"lower_ic"+str(i)+".tif")
    ic_upper.save(output_dir2+"upper_ic"+str(i)+".tif")

    


quit()
plt.figure()
figure(figsize=(10, 10), dpi=80)
plt.scatter(data3, data4, s =1)
plt.title("Weight of Independent Component 5 (top) vs. 9 (bottom)")
plt.savefig(output_dir+"weight.jpg")

plt.figure()
figure(figsize=(20, 10), dpi=80)

for i in range(50):
    ax1 = plt.subplot(1,2,2)
    ax2 = plt.subplot(2,2,1)
    ax3 = plt.subplot(2,2,3)
    data3 = pd.DataFrame(X_ica).T.iloc[i]
    data4 = pd.DataFrame(X_ica_s).T.iloc[i]
    ax2.imshow(data1[i].reshape(height, width), cmap="binary")
    ax3.imshow(data2[i].reshape(height, width), cmap="binary")
    ax2.set_title("Upper Independent Component " + str(i), fontsize=12)
    ax3.set_title("Lower Independent Component " + str(i), fontsize=12)
    ax1.scatter(data3, data4,s=1)
    ax1.set_title("Weight of Independent Component "+str(i))

    plt.savefig(output_dir+"IC_compare"+str(i)+".jpg")
    plt.clf()
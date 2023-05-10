import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#upper
#signal = [ 0,3,9,10,23,25,39,43,44,47,51]
blacklist = [3]
#lower
#signal = [0,7,8,9,11,14,25,34,37,39,50]
#signal = [0,8,9,11,14,25,34,37,39,50]
#images_dir = "/media/sf_research/cass/code/data/"
images_dir1 = "/media/sf_BIGTOP_data/1112/"
#images_dir1 = "/Users/menglin/Library/CloudStorage/Dropbox/BIGTOP data/1112/"
images_dir2 = "/media/sf_BIGTOP_data/1113/"
#images_dir2 = "/Users/menglin/Library/CloudStorage/Dropbox/BIGTOP data/1113/"
output_dir = "/media/sf_research/cass/code/output/230329upper_recon/"
#output_dir = "/Users/menglin/Library/CloudStorage/Dropbox/uva/research/cass/code/output/230405test_upper/"

left = 220
top = 520
right = 530
bottom = 670
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
    imarray = imarray.flatten()
    im_list.append(imarray)    

X_scaled = pd.DataFrame(im_list)
#print(X.head)

#data standardization 
#scaler = StandardScaler()
#scaler.fit(X)
#X_scaled = scaler.transform(X)

pca = PCA(n_components = None)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

#print(X_pca)

#signal_weight = []
#signal_pc = []

signal_weight= pd.DataFrame(X_pca)
signal_pc= pd.DataFrame(pca.components_)

signal_weight = signal_weight.iloc[:,:100]
signal_pc = signal_pc.iloc[:100,:]

signal_pc = signal_pc.T.drop(columns=blacklist, axis = 1).T
signal_weight = signal_weight.drop(columns=blacklist, axis = 1)
print(signal_weight.head)
print(signal_pc.head)

data1 = np.array(signal_weight.dot(signal_pc) + pca.mean_)
data2 = im_list

plt.figure()
for i in range(100):
    plt.clf()
    fig, axis = plt.subplots(1, 2,figsize=(20,10))
    axis[0].imshow(data2[i].reshape(height, width), cmap="binary")
    axis[1].imshow(data1[i].reshape(height, width), cmap="binary")
    axis[0].set_title("Original Image " + str(i), fontsize=12)
    axis[1].set_title("Reconstructed Image" + str(i), fontsize=12)
    plt.savefig(output_dir+"0upper_image"+str(i)+"_4.jpg")

quit()
for i in range(len(data1)):
    data1[i] = (data1[i]-np.min(data1[i]))/(np.max(data1[i])-np.min(data1[i]))
    #data2[i] = (data2[i]-np.min(data2[i]))/(np.max(data2[i])-np.min(data2[i]))
    ic_lower = Image.fromarray(data1[i].reshape(height,width),"F")
    #ic_upper = Image.fromarray(np.uint8((data2[i].reshape(height,width))*255),"L")
    ic_lower.save(output_dir+"upper_image"+str(i)+".tif")
    #ic_upper.save(output_dir2+"uppwe_pc"+str(i)+".tif")
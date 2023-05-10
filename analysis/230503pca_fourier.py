import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


images_dir = "/media/sf_research/cass/code/output/230503fourier_tif/"
output_dir = "/media/sf_research/cass/code/output/230503fourier_pc/"
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
for file in os.listdir(images_dir):
    if file.endswith("tiff"):
        images_path.append(os.path.join(images_dir,file))


im_list = []
for image in images_path:
    im = Image.open(image)
    
    #im_cropped.save(images_dir+"0.tiff")  #check the cropped area, save in tiff
    imarray = np.array(im)
    imarray = imarray.flatten()
    im_list.append(imarray)    

X_scaled = pd.DataFrame(im_list)
#print(X.head)

#data standardization 
#scaler = StandardScaler()
#scaler.fit(X)
#X_scaled = scaler.transform(X)

pca = PCA(n_components = 50)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

#print(X_pca)

#signal_weight = []
#signal_pc = []
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

c=1
data1 = pca.components_
for data in data1:
    data = (data-np.min(data))/(np.max(data)-np.min(data))
    ic_lower = Image.fromarray(data.reshape(height,width),"F")
    ic_lower.save(output_dir+"pc"+str(c)+"_fourier.tiff")
    c=c+1


quit()
for i in range(len(data1)):
    data1[i] = (data1[i]-np.min(data1[i]))/(np.max(data1[i])-np.min(data1[i]))
    #data2[i] = (data2[i]-np.min(data2[i]))/(np.max(data2[i])-np.min(data2[i]))
    ic_lower = Image.fromarray(data1[i].reshape(height,width),"F")
    #ic_upper = Image.fromarray(np.uint8((data2[i].reshape(height,width))*255),"L")
    ic_lower.save(output_dir+"upper_image"+str(i)+".tif")
    #ic_upper.save(output_dir2+"uppwe_pc"+str(i)+".tif")
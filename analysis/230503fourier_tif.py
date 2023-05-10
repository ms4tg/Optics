import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
import imageio


#images_dir = "/media/sf_research/cass/code/data/stripe/"
images_dir1 = "/media/sf_BIGTOP_data/1112/"
images_dir2 = "/media/sf_BIGTOP_data/1113/"
#output_dir ="/media/sf_research/cass/code/output/230503fourier_tif/"
output_dir ="/media/sf_research/cass/code/output/230510fourier_upper_tif/"

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



c=1
for image in images_path:
    im = Image.open(image)
    im = im.rotate(angle)
    im_cropped = im.crop((left,top,right,bottom))
    #im_cropped.save(images_dir+str(c)+".tiff")
    imarray = np.array(im_cropped)
    fourier_im = np.fft.fftshift(np.fft.fft2(imarray))

    #plt.savefig(output_dir+str(c)+"_fourier"+".jpg")

    #plt.clf()

    fourier_im[40:50,135:145]=1
    fourier_im[95:105,130:145]=1
    fourier_im[45:57,166:180]=1
    fourier_im[105:115,160:171]=1
    fourier_im[100:110,100:110]=1
    fourier_im[40:50,200:210]=1
    fourier_im[74:78,144:147]=1
    fourier_im[74:78,164:167]=1

    #fourier_im[75:78,154:156]=1
    
    data1 = abs(np.fft.ifft2(fourier_im))
    data1 = data1.astype('float32')
    data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
    #fig, axis = plt.subplots(1, 2,figsize=(20,10))
    ic_lower = Image.fromarray(data1,"F")
    ic_lower.save(output_dir+str(c)+"_fourier.tiff")
    #imageio.imwrite(output_dir+str(c)+"__fourier.tiff", [[0,255],[255,0]])
    #matplotlib.image.imsave(output_dir+str(c)+"_fourier.tiff",abs(np.fft.ifft2(fourier_im)))
    c = c+1
 
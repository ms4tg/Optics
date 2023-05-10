import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist


images_dir = "/media/sf_research/cass/code/data/image/"
#images_dir = "/media/sf_BIGTOP_data/1112/"
output_dir ="/media/sf_research/cass/code/output/230425fourier_full/"
#output_dir ="/media/sf_research/cass/code/output/230412fourier/"

left = 110
top = 290
right = 950
bottom = 900
angle = 0

pixels = (right-left)*(top-bottom)
height = top-bottom 
width = right-left 

#2d image array (height, width)
#each image is stored row after row in a 1d array
images_path = []
for file in os.listdir(images_dir):
    if file.endswith("tif"):
        images_path.append(os.path.join(images_dir,file))



c=1
for image in images_path:
    im = Image.open(image)
    im = im.rotate(angle)
    im_cropped = im.crop((left,top,right,bottom))
    im_cropped.save(images_dir+str(c)+".tiff")
    imarray = np.array(im_cropped)
    fourier_im = np.fft.fftshift(np.fft.fft2(imarray))

    fig, axis = plt.subplots(2, 2,figsize=(20,20), dpi=80)
    axis[0][1].imshow(imarray, cmap="gray")
    axis[0][0].imshow(np.log(abs(fourier_im)), cmap="gray")
    axis[0][1].set_title("Original Image " + str(c), fontsize=12)
    axis[0][0].set_title("Fourier Image" + str(c), fontsize=12)
    #plt.savefig(output_dir+str(c)+"_fourier"+".jpg")

    #plt.clf()

    fourier_im[320:370,270:330]=1
    fourier_im[395:410,270:330]=1
    fourier_im[315:325,195:205]=1
    fourier_im[250:300,520:570]=1
    fourier_im[320:325,440:445]=1
    #fourier_im[40:50,200:210]=1
    #fourier_im[74:78,144:147]=1
    #fourier_im[74:78,164:167]=1

    #fourier_im[75:78,154:156]=1

    #fig, axis = plt.subplots(1, 2,figsize=(20,10))
    axis[1][1].imshow(abs(np.fft.ifft2(fourier_im)), cmap="gray")
    axis[1][0].imshow(np.log(abs(fourier_im)), cmap="gray")
    axis[1][1].set_title("Original Image " + str(c)+" edited" , fontsize=12)
    axis[1][0].set_title("Fourier Image "+ str(c)+" edited" , fontsize=12)
    plt.savefig(output_dir+str(c)+"_fourier"+".jpg")
    plt.clf()
    c = c+1
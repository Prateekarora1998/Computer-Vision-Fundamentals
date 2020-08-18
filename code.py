import numpy as np
from numpy import *
from random import randrange
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.misc import imsave
from skimage.color import rgb2gray
from skimage.transform import rescale,resize
from scipy import ndimage, misc
from scipy.ndimage import gaussian_filter

# Task - 1

a = np.array([[2, 4, 5],[5, 2, 200]])
b = a[0 , :]
f = np.random.randn(500,1)
g = f[ f < 0 ]
x = np.zeros (100) + 0.35
y = 0.6 * np.ones([1, len(x)])
z = x - y
a = np.linspace(1,200)
b = a[: :-1]
b[b <=50]=0

# Task - 2

## 1)

image = cv2.imread('Lenna.png')
grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Inbuilt function converting colored image to gray image

x_s=image[:,:,0]
y_s=np.shape(x_s)
zeros_array=np.zeros(y_s)
zeros_array =255- x_s

img_conct = cv2.hconcat((grayscale_image,zeros_array))
cv2.imshow("Grayscale and Negative Image Simultaneously",img_conct)
cv2.waitKey(0)

## 2)

flip_image = cv2.flip(grayscale_image,0)
cv2.imshow("Flipped Image",flip_image)
cv2.waitKey(0)

## 3)

old_image = image
blue_color_pixels = old_image[:,:,2].copy() # Copying the blue channel of the old image and storing it into a variable
red_color_pixels = old_image[:,:,0].copy() # Copying the red channel of the old image and storing it into a variable
old_image[:,:,0] = blue_color_pixels # Swapping red channel with blue channel
old_image[:,:,2] = red_color_pixels # Swapping blue channel with red channel
cv2.imshow("Red and blue channels exchanged",old_image)
cv2.waitKey(0)

## 4)

average_image = (1/2 * (flip_image + grayscale_image)).astype(np.uint8) # astype is used to convert the values into integer type as after average of the values, it becomes float type
cv2.imshow("Average Image",average_image)
cv2.waitKey(0)

## 5)

random_values = np.random.randint(low = 0, high = 255, size = (512,512)) # Generating random values
new_image = grayscale_image + random_values  ## Adding the random values in the grayscale image
new_image = np.clip(new_image,0,255).astype(np.uint8) ## Clipping the image
cv2.imshow("Clipped Image",new_image)
cv2.waitKey(0)

# Task - 3

## 1)

image1 = cv2.imread('Image_1.jpeg')
image2 = cv2.imread('Image_2.jpeg')
image3 = cv2.imread('Passport_pic.jpg')

## 2)

image1 = cv2.resize(image1,(1024,720))
image2 = cv2.resize(image2,(1024,720))
image3 = cv2.resize(image3,(1024,720))
plt.imsave('face01_u6742441.jpeg',image1)
plt.imsave('face02_u6742441.jpeg',image2)
plt.imsave('face03_u6742441.jpeg',image3)

## 3)

### a)

image1_new = cv2.imread('face03_u6742441.jpeg')
image1_new = cv2.resize(image1_new,(768,512))
plt.imshow(image1_new)
plt.imsave('New_pixel_image.jpeg',image1_new)
plt.show()

### b)

red_image = image1_new[:,:,0]
cv2.imshow("Red Gray Channel Plot",red_image)
cv2.waitKey(0)
cv2.imwrite('Red_grayscale_image.jpeg',red_image)

green_image = image1_new[:,:,1]
cv2.imshow("Green Gray Channel Plot",green_image)
cv2.waitKey(0)
cv2.imwrite('Green_grayscale_image.jpeg',green_image)

blue_image = image1_new[:,:,2]
cv2.imshow("Blue Gray Channel Plot",blue_image)
cv2.waitKey(0)
cv2.imwrite('Blue_grayscale_image.jpeg',blue_image)

### c)

plt.hist(red_image.ravel(),256,[0,256])
plt.ylabel("Frequency")
plt.xlabel("RGB Pixels")
plt.title("Histogram of red gray scale image")
plt.show()

plt.hist(green_image.ravel(),256,[0,256])
plt.ylabel("Frequency")
plt.xlabel("RGB Pixels")
plt.title("Histogram of green gray scale image")
plt.show()

plt.hist(blue_image.ravel(),256,[0,256])
plt.ylabel("Frequency")
plt.xlabel("RGB Pixels")
plt.title("Histogram of blue gray scale image")
plt.show()

### d)

img = cv2.imread('New_pixel_image.jpeg')
src0 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Converting colored image to gray scale
equ = cv2.equalizeHist(src0) #This function equalizes the histogram formed
plt.title("Image after Histogram Equalization")
plt.imshow(equ)
plt.show()

img = cv2.imread('Red_grayscale_image.jpeg')
src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting colored image to gray scale
equ = cv2.equalizeHist(src) #This function equalizes the histogram formed
plt.title("Red channel of image after Histogram Equalization")
plt.imshow(equ,cmap = "gray")
plt.show()

img1 = cv2.imread('Green_grayscale_image.jpeg')
src1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # Converting colored image to gray scale
equ = cv2.equalizeHist(src1) #This function equalizes the histogram formed
plt.title("Green channel of image after Histogram Equalization")
plt.imshow(equ,cmap = "gray")
plt.show()

img2 = cv2.imread('Blue_grayscale_image.jpeg')
src2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # Converting colored image to gray scale
equ = cv2.equalizeHist(src2) #This function equalizes the histogram formed
plt.title("Blue channel of image after Histogram Equalization")
plt.imshow(equ, cmap = "gray")
plt.show()

# Task - 4 

## 1)

read_image = cv2.imread('face03_u6742441.jpeg')

fig_size = plt.figure(figsize=(8,5))
ax1 = fig_size.add_subplot(121)
ax2 = fig_size.add_subplot(122)

crop_img = read_image[0:500, 250:750] # Cropping the image
new_crop_image = cv2.resize(crop_img,(256,256)) # Resizing the image
grayscale_crop_image = cv2.cvtColor(new_crop_image, cv2.COLOR_RGB2GRAY) # Converting colored image to gray scale
plt.imsave('Crop_256_Grayscale_image.png',grayscale_crop_image)

ax1.set_title("RGB Image")
ax1.imshow(new_crop_image)

ax2.set_title("Grayscale Image")
ax2.imshow(grayscale_crop_image, cmap = 'gray')

plt.show()
## 2)

new_random_values_gen = (grayscale_crop_image + (15 * np.random.randn(256,256))) # Adding the random noise
new_random_values_gen = cv2.normalize(new_random_values_gen,new_random_values_gen, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Normalizing the image and the converting the float type to int type using astype function
plt.title("Grayscale Image with added noise")
plt.imshow(new_random_values_gen, cmap = 'gray')
plt.show()
## 3)

fig_size = plt.figure(figsize=(10,5))
ax1 = fig_size.add_subplot(121)
ax2 = fig_size.add_subplot(122)

ax1.hist(new_crop_image.ravel(),256,[0,256]) # Displays the original histogram
ax1.set_ylabel("Frequency")
ax1.set_xlabel("RGB Pixels")
ax1.set_title("Histogram of cropped image")

ax2.hist(new_random_values_gen.ravel(),256,[0,256]) # Displays the histogram with added noise
ax2.set_ylabel("Frequency")
ax2.set_xlabel("RGB Pixels")
ax2.set_title("Histogram of cropped image with noise")
plt.show()

## 4)

def my_Gauss_filter(noise_image,my_gaussian_kernel):
    m = len(noise_image[:,0]) # Getting the shape of x axis
    n = len(noise_image[:,1]) # Getting the shape of y axis
    output = np.zeros((m,n))
    for i in range(0,m-2):
        for j in range(0,n-2):
            sum_values = 0
            for k in range(-2,3):
                for l in range(-2,3):
                    sum_values = sum_values + np.multiply(noise_image[i+k,j+l],my_gaussian_kernel[2+k,2+l]) # Muliptly the kernel and the noise image for the convolution
            output[i][j] = sum_values         
    return output

## 5)

def guassian_kernel(sigma):
    kernel = np.zeros((5,5))
    rows_kernel = len(kernel[0])//2
    colunms_kernel = len(kernel[1])//2
    pi_square = (np.pi)**2
    
    for i in range(-rows_kernel,rows_kernel+1):
        for j in range(-colunms_kernel, colunms_kernel+1):
            kernel[i+rows_kernel][j+colunms_kernel] = ((1/sigma*2*pi_square))*(np.exp(-(i*i + j*j)/(2*sigma**2))) # Using the formula for the gaussian kernel
            
    return kernel

kernel = guassian_kernel(2) # Setting sigma as 2
output = my_Gauss_filter(new_random_values_gen,kernel)
plt.title("Image gone through self made gaussain filter_1")
plt.imshow(output,cmap = 'gray')
plt.show()

kernel1 = guassian_kernel(3) # Setting sigma as 3
output1 = my_Gauss_filter(new_random_values_gen,kernel1)
plt.title("Image gone through self made gaussain filter_2")
plt.imshow(output1,cmap = 'gray')
plt.show()

kernel2 = guassian_kernel(4) # Setting sigma as 4
output2 = my_Gauss_filter(new_random_values_gen,kernel2)
plt.title("Image gone through self made gaussain filter_3")
plt.imshow(output2,cmap = 'gray')
plt.show()
## 6)

new_output_formed = cv2.GaussianBlur(new_random_values_gen,(5,5),cv2.BORDER_DEFAULT) # Using the inbuilt function given by cv

fig_size = plt.figure(figsize=(10,5))
ax1 = fig_size.add_subplot(121)
ax2 = fig_size.add_subplot(122)

ax1.set_title("Image gone through predefined gaussain filter")
ax1.imshow(new_output_formed,cmap = "gray")

ax2.set_title("Image gone through self made gaussain filter_1")
ax2.imshow(output,cmap="gray")
plt.show()

# Task - 5

def my_sobel_detector(my_image,x_kernel,y_kernel):
    m = len(my_image)
    n = len(my_image[0])
    S1 = np.zeros((m,n))
    sum_values = 0
    sum_values1 = 0
    sum_values2 = 0
    for i in range(1,m-2):
        for j in range(1,n-2):
            sum_values = np.sum(np.multiply(my_image[i-1:i+2,j-1:j+2] , x_kernel))
            sum_values1 = np.sum(np.multiply(my_image[i-1:i+2,j-1:j+2] , y_kernel))
            sum_values2 = np.sqrt(sum_values*sum_values + sum_values1*sum_values1) 
            S1[i][j] = sum_values2
    return S1

kernel = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])
kernel1 = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])
output_sobel = my_sobel_detector(grayscale_crop_image,kernel,kernel1)

sobelx = cv2.Sobel(grayscale_crop_image,cv2.CV_64F,1,0,ksize=3) # Using the inbuilt function
sobely = cv2.Sobel(grayscale_crop_image,cv2.CV_64F,0,1,ksize=3) # Using the inbuilt function
sobel_new = np.sqrt(sobelx*sobelx + sobely*sobely)

fig_size = plt.figure(figsize=(10,5))
ax1 = fig_size.add_subplot(121)
ax2 = fig_size.add_subplot(122)

ax1.set_title("Self made sobel filter applied on Image")
ax1.imshow(output_sobel, cmap = "gray")

ax2.set_title("Predefined sobel filter applied on Image")
ax2.imshow(sobel_new, cmap = "gray")

plt.show()
# Task - 6

## 1)

image_new_resize = cv2.resize(image1_new,(512,512))
plt.imshow(image_new_resize)

def rotation(image, degree):
    h, w = image.shape[:2]
    output_Image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            x = (i - int(h/2))*math.cos(math.radians(degree)) - ( j - int(w/2))*math.sin(math.radians(degree))+ int(h/2)
            y = (i - int(h/2))*math.sin(math.radians(degree)) + (j- int(w/2))*math.cos(math.radians(degree))+ int(w/2)
            if 0 < x < w and 0 < y < h:
                output_Image[i, j] = image[int(x), int(y)]
    return output_Image

rotation_minus_90_degree = rotation(image_new_resize,-90)
rotation_minus_45_degree = rotation(image_new_resize,-45)
rotation_minus_15_degree = rotation(image_new_resize,-15)
rotation_45_degree = rotation(image_new_resize,45)
rotation_90_degree = rotation(image_new_resize,90)

plt.title("Image rotated in 90 degree anticlockwise")
plt.imshow(rotation_minus_90_degree)
plt.show()

plt.title("Image rotated in 45 degree anticlockwise")
plt.imshow(rotation_minus_45_degree)
plt.show()

plt.title("Image rotated in 15 degree anticlockwise")
plt.imshow(rotation_minus_15_degree)
plt.show()

plt.title("Image rotated in 45 degree clockwise")
plt.imshow(rotation_45_degree)
plt.show()

plt.title("Image rotated in 90 degree clockwise")
plt.imshow(rotation_90_degree)
plt.show()

## 3)

## Bilinear interpolation
bilinear_img = new_crop_image[100:120,80:100]
rows_1,cols_1 = bilinear_img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols_1/2,rows_1/2),0,1) 
result_linear = cv2.warpAffine(bilinear_img,rotation_matrix,(cols_1,rows_1),flags= cv2.INTER_LINEAR)
plt.imshow(result_linear)
plt.show()

## Nearest Interpolation
near_img = new_crop_image[100:120,80:100]
rows,cols = near_img.shape[:2]
rotation_matrix_1 = cv2.getRotationMatrix2D((cols/2,rows/2),0,1) 
result_near = cv2.warpAffine(near_img,rotation_matrix_1,(cols,rows),flags= cv2.INTER_NEAREST)
plt.imshow(result_near)
plt.show()
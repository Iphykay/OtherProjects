import numpy as np
import  cv2 as cv
import matplotlib.pyplot as plt


image = "C:/Users/Starboy/OneDrive - rit.edu/Courses/IPCV/Assignments/HW3/HD-wallpaper-scorpion-mortal-kombat-sadecekaan.jpg"

img = cv.imread(image)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.title('Original Image')
plt.imshow(img, cmap='gray')


# Separating the image into 3bands
red_img = img[:,:,0]
green_img = img[:,:,1]
blue_img = img[:,:,2]
plt.rcParams["figure.figsize"] = [10,15]; Channels = ['Red', 'Green', 'Blue']
fig, plots = plt.subplots(ncols=3, nrows=1)
for i, subplot in zip(range(3), plots):
    Img = np.zeros(img.shape)
    Img[:,:,i] = img[:,:,i]
    # normalize between 0 and 1
    Img = Img/255
    subplot.set_title(Channels[i])
    subplot.imshow(Img)


# Flatten the bands
red_img = red_img.flatten()
green_img = green_img.flatten()
blue_img = blue_img.flatten()


# Finding the means of the Individual Bands
Red_mean = red_img.mean()
Green_mean = green_img.mean()
Blue_mean = blue_img.mean()


# Mean Subtracted Bands
Red_mean_subtract = red_img-Red_mean
Green_mean_subtract = green_img-Green_mean
Blue_mean_subtract = blue_img-Blue_mean


# Stacking them all Up
A = np.vstack([Red_mean_subtract,Green_mean_subtract,Blue_mean_subtract]).T

# Finding the Covariance Matrix
n = red_img.shape[0]

S = (1.0/(n-1))*np.dot(A.T,A)


# Finding the EigenValues and EigenVectors
EigenValues, EigenVectors = np.linalg.eig(S)
EigenValues = np.diag(EigenValues)

#Principle Components Vectors
PC1 = -np.dot(A,EigenVectors[:,0]).reshape(img.shape[:2])
PC2 = np.dot(A,EigenVectors[:,1]).reshape(img.shape[:2])
PC3 = np.dot(A,EigenVectors[:,2]).reshape(img.shape[:2])

PCImages = [PC1, PC2, PC3]; Name = ['PC1', 'PC2', 'PC3']
def PC_Images(images):
   plt.figure(figsize=(10,10))
   for i in range(0, len(images)):
       plt.subplot(1, 3, i+1)
       plt.title(Name[i])
       plt.imshow(images[i], cmap='gray')
   plt.show()

PC_Visuals = PC_Images(PCImages)

# The percentage of information (variance) retained if the data is projected onto PC Vectors
print(EigenValues[0,0] / np.sum(EigenValues))
print((EigenValues[0,0] + EigenValues[1,1]) / np.sum(EigenValues))
print((EigenValues[0,0] + EigenValues[1,1] + EigenValues[2,2]) / np.sum(EigenValues))




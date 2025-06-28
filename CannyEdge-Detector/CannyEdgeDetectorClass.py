import numpy as np
import cv2
import matplotlib.pyplot as plt


image_path = "C:/Users/Starboy/OneDrive - rit.edu/Courses/Grad Lab/GRAD-LAB PROJECT 2/womanpicture.jpg"

class CannyDetector:
    """
    A popular edge detection algorithm used in image processing
    """
    def __init__(self, image):
        self.image = image

    def readimage(self, image):
        """
        This method read the file and converts it to gray-scale.
        """

        image = self.image
        read_image = cv2.imread(image, 0)

        return read_image
    
    def spatial_2d_convolution(self, image, kernel):
        """
        This function will implement the two dimentional spatial convolution
        kernel: the receptive field that does the averaging on the image
        """
        import numpy as np
        m, n = kernel.shape
        if (m == n):
            col_y, col_x = image.shape
            col_y = col_y - m + 1
            col_x = col_x - m + 1
            Conv_Filter = np.zeros((col_y,col_x))
            for i in range(col_y):
                for j in range(col_x):
                    Conv_Filter[i,j] = np.sum(image[i:i+m, j:j+m]*kernel)
        return Conv_Filter
    
    def gaussian_kernel(self, size, sigma=1):
        """
        This method does the Gaussian filter on images.
        sigma: it determines the spread of the distribution used in smoothing or blurring
        """
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def sobel_filters(self, img):
        """
        This method computes the sobel filters on the x and y direction
        """
        dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
        Ix = self.spatial_2d_convolution(img, dx)
        Iy = self.spatial_2d_convolution(img, dy)
    
        G = np.hypot(Ix, Iy)
        G = (G /np.max(G)) * 255
        theta = np.arctan2(Iy, Ix)
    
        return G, theta
    
    def non_max_suppression(self, image, D):
        M, N = image.shape
        Z = np.zeros((M,N), dtype=np.int32)
        for i in range(1,M-1):
            for j in range(1,N-1):
                if (D[i,j]>=-22.5 and D[i,j]<=22.5) or (D[i,j]<-157.5 and D[i,j]>=-202):
                    if (image[i,j] >= image[i,j+1]) and (image[i,j] >= image[i,j-1]):
                        Z[i,j] = image[i,j]
                    else:
                        Z[i,j] = 0
                elif (D[i,j]>=22.5 and D[i,j]<=67.5) or (D[i,j]<-112.5 and D[i,j]>=-157.5):
                    if (image[i,j] >= image[i+1,j+1]) and (image[i,j] >= image[i-1,j-1]):
                        Z[i,j] = image[i,j]
                    else:
                        Z[i,j] = 0
                elif (D[i,j]>=67.5 and D[i,j]<=112.5) or(D[i,j]<-67.5 and D[i,j]>=-112.5):
                    if (image[i,j] >= image[i+1,j]) and (image[i,j] >= image[i-1,j]):
                        Z[i,j] = image[i,j]
                    else:
                        Z[i,j] = 0
                elif (D[i,j]>=112.5 and D[i,j]<=157.5) or (D[i,j]<-22.5 and D[i,j]>=-67.5):
                    if (image[i,j] >= image[i+1,j-1]) and (image[i,j] >= image[i-1,j+1]):
                        Z[i,j] = image[i,j]
                    else:
                        Z[i,j] = 0
        return Z
    
    def threshold(self, img, LowThreRatio=0.05, HighThresRatio=0.09):
        """
        This method does thresholding on the pixels from the No_max suppression.
        LowThreRatio is the Low Threshold Ratio and HighThreRatio is the High Threshold Ration
        """
        highThreshold = np.max(img) * HighThresRatio
        lowThreshold = highThreshold * LowThreRatio
    
        M, N = img.shape
        res = np.zeros((M,N), dtype=np.float32)
    
        weak = np.int32(25)         # the values can be cahnge to see they differ
        strong = np.int32(255)
    
        strong_i, strong_j = np.where(img >= highThreshold)
    
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
    
        return res, weak, strong
    
    def hysteresis(self, image, weak, strong=255):
        M, N = image.shape  
        img = image.copy()
        for i in range(1, M):
            for j in range(1, N):
                if (img[i,j] == weak):
                    if ((img[i, j+1] == strong) or (img[i, j-1] == strong) or (img[i-1, j] == strong)
                        or (img[i+1, j] == strong) or (img[i-1, j-1] == strong)
                        or (img[i+1, j-1] == strong) or (img[i-1, j+1] == strong) or (img[i+1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
    
        img2 = image.copy()
        for i in range(M - 1, 0, -1):
            for j in range(N - 1, 0, -1):
                if img2[i, j] == weak:
                    if ((img2[i, j + 1]) == 255 or (img2[i,j-1] == 255) or (img2[i-1,j] == 255) 
                        or (img2[i+1,j] == 255) or (img2[i-1,j-1] == 255) or (img2[i + 1, j - 1] == 255) 
                        or (img2[i,j+1] == 255) or (img2[i+1,j+1] == 255)):
                        img2[i, j] = 255
                    else:
                        img2[i, j] = 0

        img3 = image.copy()
        for i in range(M-1,0,-1):
            for j in range(1, N):
                if img3[i,j] == weak:
                    if ((img3[i,j+1] == 255) or (img3[i,j-1] == 255) or (img3[i-1,j] == 255) 
                        or (img3[i+1,j] == 255) or (img3[i-1,j-1] == 255) or (img3[i+1,j-1] == 255)
                        or (img3[i-1,j+1] == 255) or (img3[i+1,j+1] == 255)):
                        img3[i,j] = 255
                    else:
                        img3[i,j] = 0
    
        img4 = image.copy()
        for i in range(1, M):
            for j in range(N - 1, 0, -1):
                if img4[i,j] == weak:
                    if ((img4[i,j+1] == 255) or img4[i,j-1] == 255 or img4[i-1,j] == 255 or img4[i+1,j] == 255 
                        or img4[i-1,j-1] == 255 or img4[i+1,j-1] == 255 or img4[i-1,j+1] == 255 or img4[i+1,j+1] == 255):
                        img4[i,j] = 255
                    else:
                        img4[i,j] = 0

        output_ = img + img2 + img3 + img4
 
        output_[output_ > 255] = 255
 
        return output_
    
    def displayimage(self, image, name:str):
        """
        This method helps in diplaying the image.
        """
        if isinstance(image, (float, int)):
            self.image = image
        output = plt.imshow(image, cmap='gray')
        output = plt.title(name)
        output = plt.show()
        return output
    
    def run(self):
        output = self.readimage(image_path)
        self.displayimage(output, 'Original Image')
        self.displayimage(self.spatial_2d_convolution(output, self.gaussian_kernel(5,1)), 'Gaussian Filter')
        self.displayimage(self.sobel_filters(output)[0], 'Sobel Gradient')
        self.displayimage(self.non_max_suppression(self.sobel_filters(output)[0],self.sobel_filters(output)[1]),'Non-max suppression')
        self.displayimage(self.threshold(self.non_max_suppression(self.sobel_filters(output)[0],self.sobel_filters(output)[1]), 
                                         LowThreRatio=0.05, HighThresRatio=0.09)[0],'Double Threshold')
        In = self.threshold(self.non_max_suppression(self.sobel_filters(output)[0],self.sobel_filters(output)[1]),LowThreRatio=0.05, HighThresRatio=0.09)[0]
        self.displayimage(self.hysteresis(In, 25),'Edge Tracking by Hysteresis')
        

def main():
    simulate = CannyDetector(image_path)
    simulate.run()

if __name__ == "__main__":
    main()
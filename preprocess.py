import cv2
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.io import imread, imshow
import numpy as np 
from skimage import filters

class PreprocessImage(object):
    def __init__(self, img):
    # read image as grayscale
        self.img = cv2.imread(img, 0)
        
    

    #img should be an 2d numpy array

    def detect_ridges(self, sigma=1.0,o='rc'):
        H_elems = hessian_matrix(self.img, sigma=sigma, order=o)
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
        self.maxima_ridges = maxima_ridges.copy()
        return maxima_ridges, minima_ridges

    def plot_images(self,*images):
        images = list(images)
        images.append(self.img)
        n = len(images)
        fig, ax = plt.subplots(ncols=n, sharey=True)
        for i, img in enumerate(images):
       
            ax[i].imshow(img, cmap='gray')
            ax[i].axis('off')
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
        plt.show()
    
    def checkImageEligibility(img):
        if img.ndim == 2:

            channels = 1 #single (grayscale)

        if img.ndim == 3:

            channels = image.shape[-1]

        return channels,img.dtype
    
    #thresholding image
    def threshold(self):
        # https://stackoverflow.com/questions/57231053/opencv-error-with-adaptive-thresholding-error-215        
        gray = cv2.convertScaleAbs(self.maxima_ridges, alpha=255/self.maxima_ridges.max())
        th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        filename = 'threshold-res.jpg'
        cv2.imwrite(filename, th3)
        return filename

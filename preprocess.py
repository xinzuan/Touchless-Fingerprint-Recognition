import cv2
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.io import imread, imshow
import numpy as np 
from skimage import filters
from PIL import Image
import fingerprint_enhancer
import math

import os


TEMP_PATH ='/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/backend/complete/src/resources/temp/'

def get_absolute_path(file_path):
    if not os.path.abspath(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    return file_path

class PreprocessImage(object):
    def __init__(self):
        super().__init__()
    def __init__(self, img):
    # read image as grayscale
        self.img = cv2.imread(img, 0)
        
    

    #img should be an 2d numpy array

    def detect_ridges(self,img, sigma=1.0,o='rc'):

        H_elems = hessian_matrix(img, sigma=sigma, order=o)
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
        self.maxima_ridges = maxima_ridges.copy()
        return img,maxima_ridges, minima_ridges

    def plot_images(self,*images):
        images = list(images)
        # images.append(self.img)
        n = len(images)
        fig, ax = plt.subplots(ncols=n, sharey=True)
        for i, img in enumerate(images):
       
            ax[i].imshow(img, cmap='gray')
            ax[i].axis('off')
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
        plt.show()
    
    def checkImageEligibility(self,img):
        if img.ndim == 2:

            channels = 1 #single (grayscale)

        if img.ndim == 3:

            channels = image.shape[-1]

        return channels,img.dtype
    
    #thresholding image
    def threshold(self,img):
        # https://stackoverflow.com/questions/57231053/opencv-error-with-adaptive-thresholding-error-215        
       # gray = cv2.convertScaleAbs(self.maxima_ridges, alpha=255/self.maxima_ridges.max())
        # th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #             cv2.THRESH_BINARY,11,2)
        # filename = os.path.join(TEMP_PATH,'threshold-res.jpg')
        # cv2.imwrite(filename, th3)
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
    

    
    #show image in PIL
    def read_finger_media(self, file_path):
        file_path = get_absolute_path(file_path)

        stego_media = Image.open(file_path)
        self.stego_media = np.array(stego_media)
        self.determine_stego_capacity()
        self.stego_media_file_name = os.path.basename(file_path)

    def determine_stego_capacity(self):
        capacity = 1
        for dimension in self.stego_media.shape:
            capacity *= dimension
        self.stego_media_shape = self.stego_media.shape
        self.stego_media_capacity = capacity
    
    #for zoom
    def clipped_zoom(self,img, zoom_factor, **kwargs):

        h, w = img.shape[:2]

        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # Zooming out
        if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

        # Zooming in
        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]

        # If zoom_factor == 1, just return the input array
        else:
            out = img
        return out

    # masking image with ellipse
    # One argument is the center location (x,y).
    #  Next argument is axes lengths (major axis length, minor axis length). 
    # angle is the angle of rotation of ellipse in anti-clockwise direction. 
    # startAngle and endAngle denotes the starting and ending of ellipse arc measured in clockwise direction from major axis. i.e. giving values 0 and 360 gives the full ellipse
    def ellipse_masking(self,img):
        # get shape
        height, width = img.shape



        # Create new blank image and shift ROI to new coordinates
        mask = np.zeros(img.shape, dtype=np.uint8)

        # x = width//2 - ROI.shape[0]//2 
        # y = height//2 - ROI.shape[1]//2 
        x = width//2
        y = height//2 
        center_coordinates = (x, y) 
        major_axis = y + 50
        minor_axis = x - 30
        axesLength = (major_axis, minor_axis) 
        
        angle = 90
        
        startAngle = 0
        
        endAngle = 360
        
        # Red color in BGR 
        color = 255 
        
        # Line thickness of 5 px 
        thickness = -1
        
        # Using cv2.ellipse() method 
        # Draw a ellipse with red line borders of thickness of 5 px 
        cv2.ellipse(mask, center_coordinates, axesLength, 
                angle, startAngle, endAngle, color, thickness)

        # img = clipped_zoom(img, 1.5)
        # imshow(img)
        # cv2.imwrite('/content/zoom.jpg', img)

        # bitwise masking 
        masked = cv2.bitwise_and(img, img, mask=mask)

        return masked
    # improve contrast using clahe
    def increase_contrast(self,img,clipLimit=3,size=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=size)
        return clahe.apply(img)
    
    # gamma correction : untuk control overall brightnes
    def gamma_correction(self,gray):
        # compute gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.mean(gray)
        gamma = math.log(mid*255)/math.log(mean)
        # print(gamma)

        # do gamma correction
        img_gamma1 = np.power(gray, gamma).clip(0,255).astype(np.uint8)
        return img_gamma1

    #adaptive threshold 
    def adaptive_threshold(self,img):
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        img=np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
        return img
    
    # preprosess all the image:
    def preprocess_image(self):
        img = self.img.copy()
        img = self.gamma_correction(img)
        img = self.increase_contrast(img)
        img = self.ellipse_masking(img)
        img = self.threshold(img)
        img,maxima_ridges,minima_ridges = self.detect_ridges(img,2)

        result = fingerprint_enhancer.enhance_Fingerprint(maxima_ridges)
        return self.img,result



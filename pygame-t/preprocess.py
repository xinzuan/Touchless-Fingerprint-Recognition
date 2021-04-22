import cv2
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.io import imread, imshow
import numpy as np 
from skimage import filters
from PIL import Image
import fingerprint_enhancer
import math
import os.path as osp
import glob
import os
from cv2 import dnn_superres
from superresolution import SuperResolution,SRESGRAN
from state import SRType,PATH
import time
import imutils
import requests
import pandas as pd
from openpyxl import load_workbook
# from gabor 
# edsr around 5 minutes
# Create an SR object
from imagequality import ImageQualityMetrics
from optparse import OptionParser

TEMP_PATH ='/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/backend/complete/src/resources/temp/'
# FSRCNN_PATH ='/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/models/FSRCNN_x4.pb'
# EDSR_PATH ='/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/models/EDSR_x4.pb'
SIZE = 32
def get_absolute_path(file_path):
    if not os.path.abspath(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    return file_path

class PreprocessImage(object):
    def __init__(self):
        self.count = 1

       
        self.fsrcnn_sr = SuperResolution(SRType.FSRCNN.value[0],PATH.FSRCNN_MODEL.value)
        self.edsr_sr = SuperResolution(SRType.EDSR.value[0],PATH.EDSR_MODEL.value)
        self.esgran_sr = SRESGRAN(PATH.ESGRAN_MODEL.value)




        super().__init__()


    #img should be an 2d numpy array

    def detect_ridges(self,img, sigma=1.0,o='xy'):

        H_elems = hessian_matrix(img, sigma=sigma, order=o)
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
        self.maxima_ridges = maxima_ridges.copy()
        return img,maxima_ridges, minima_ridges

    def plot_images(self,*images):
        images = list(images)
        # images.append()
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
    def threshold(self,image, w=SIZE):
        # https://stackoverflow.com/questions/57231053/opencv-error-with-adaptive-thresholding-error-215        
        # img = cv2.convertScaleAbs(img, alpha=255/img.max())
        # th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #             cv2.THRESH_BINARY,11,2)
        # filename = os.path.join(TEMP_PATH,'threshold-res.jpg')
        # cv2.imwrite(filename, th3)
        # image = np.copy(img)
        height, width = image.shape
        for y in range(0, height, w):
            for x in range(0, width, w):
                block = image[y:y+w, x:x+w]
                
                
                
                threshold = np.average(block)

                if threshold == 255:
                    image[y:y+w, x:x+w] = 120
                # elif threshold >=3.0 and threshold<=10.0:
                #      image[y:y+w, x:x+w] = 255
                # elif threshold == 255.0:
                #     pass
                # # elif threshold >=4 and threshold<5:
                # #      image[y:y+w, x:x+w] = 100
                # elif threshold >=127 and threshold<128:
                #      image[y:y+w, x:x+w] = np.where(((block >= 126) & (block<128)) ,255, 0)
                # elif threshold <100:
                #      image[y:y+w, x:x+w] = np.where(block < threshold ,255, 0)
                # else:
                #     # print(threshold)
                    
                #     # image[y:y+w, x:x+w] = np.where(((block >= 127) & (block<=threshold)) ,255, 0)
                #     image[y:y+w, x:x+w] = np.where(block <=threshold ,0, 255)
                    
                    
                    
                # image[y:y+w, x:x+w] = np.where(block <=threshold, 255, 0)

        return image
        # return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #             cv2.THRESH_BINARY,11,2)

    
    def findMask(self,image, threshold=0.1, w=SIZE):


        mask = np.empty(image.shape)
        height, width = image.shape
        height -=w*2
        width -=w
        for y in range(0, height, w):
            for x in range(0, width, w):
                block = image[y:y+w, x:x+w]
                standardDeviation = np.std(block)
                if standardDeviation < threshold:
                    mask[y:y+w, x:x+w] = 0.0
                elif block.shape != (w, w):
                    mask[y:y+w, x:x+w] = 0.0
                else:
                    mask[y:y+w, x:x+w] = 1.0

        return mask
    



    
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

    # masking image with rectangle
    # One argument is the center location (x,y).
    #  Next argument is axes lengths (major axis length, minor axis length). 
    # angle is the angle of rotation of rectangle in anti-clockwise direction. 
    # startAngle and endAngle denotes the starting and ending of rectangle arc measured in clockwise direction from major axis. i.e. giving values 0 and 360 gives the full rectangle
    def get_points(self,img):


        # get shape
       
        height, width = img.shape
        

       

        

        (y, x) = np.where(img == 255)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = ((np.max(y)), np.max(x))

        
       


        

        return bottomx,bottomy,topx,topy
    # improve contrast using clahe
    def increase_contrast(self,img,clipLimit=3,size=(16,16)):
        # if img.ndim == 2:
        #     print(img.shape)
        #     img = self.to_rgb(img)
        clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=size)
        # imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # y, u, v = cv2.split(imgYUV)
        
        # c_y = clahe.apply(y)
        # res =cv2.merge((y,u,v))
        # out = cv2.cvtColor(res,cv2.COLOR_YUV2BGR)
        out = clahe.apply(img)
        return out
    def increase_contrast_sr(self,img,clipLimit=4,size=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=size)
        return clahe.apply(img)
    # gamma correction : untuk control overall brightnes, gelap -> terang
    def gamma_correction(self,gray,threshold=127):
        # compute gamma = log(mid*255)/log(mean)
        mean = np.mean(gray)
        
        # print(mean)
        if mean > threshold:
            return gray 
        else:
        
            mid = 0.5
            
            gamma = math.log(mid*255)/math.log(mean)
            # print(gamma)

            # do gamma correction
            img_gamma1 = np.power(gray, gamma).clip(0,255).astype(np.uint8)

            # print(img_estim(img_gamma1))
            return img_gamma1

    def automatic_brightness_and_contrast(self,image, clip_hist_percent=1):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = image.copy()

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist) 

        # Calculate grayscale histogram
        hist = cv2.calcHist([image],[0],None,[256],[0,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return auto_result

    #adaptive threshold 
    def adaptive_threshold(self,img):
        # img = self.automatic_brightness_and_contrast(img)
        # img_name = "_BEFORE THRES_{}.png".format(self.count)
        # cv2.imwrite(img_name,img)
        # img = self.to_gray(img)
        # img = self.normalize(img)
        img = cv2.convertScaleAbs(img, alpha=255/img.max(),beta=img.min())
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        
       
        return img
    def adaptive_threshold_sr(self,img):
        # img = self.normalize(img)
        # img = cv2.convertScaleAbs(img, alpha=255/img.max())
       
        
       
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    def otsu_thresholding(self,img):
        otsu_threshold,img =  cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
       
        return img
    
    def normalize_image(self,img):
        return np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

    # convert to gray 
    def to_gray(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def to_rgb (self,img):
        return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    #Segmentation finger only
    def segment_finger(self,img):
        
        gray = self.to_gray(img)
        img = self.smoothing(img)
        

        imgYCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        #Splitting into YCbCr
        Y, Cb, Cr = cv2.split(imgYCrCb)

        # img_merge =cv2.cvtColor(img_merge,cv2.COLOR_YCrCb2RGB)
        # img_merge = self.to_gray(img_merge)
        img = self.to_gray(img)
        binary_mask = self.otsu_thresholding(img)
        # # copy the image to create a binary mask later
        # binary_mask = np.copy(imgYCrCb)


        # for i in range(imgYCrCb.shape[0]):
        #     for j in range(imgYCrCb.shape[1]):
            
        #         if Cr[i, j] >= 133 and Cr[i,j] <=180 and Cb[i,j] >= 77 and Cb[i,j] <= 127:
        #             # paint it white (finger region)
        #             binary_mask[i, j] = [255, 255, 255]
        #         else:
        #             # paint it black 
        #             binary_mask[i, j] = [0, 0, 0]
        
        mask = binary_mask.copy()
        # mask = self.to_gray(mask)
        
        
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            # grab the largest contour, then draw a mask for the pill
            c = max(cnts, key=cv2.contourArea)
            mask = np.zeros(gray.shape, dtype="uint8")
            mask = cv2.drawContours(mask, [c], -1, 255, -1)

      
        # max_height = (y+h)*0.6
        # max_width = (x+w)-15
        # start_x = x+15
        # bottomx,bottomy,topx,topy = self.get_points(mask)
        # # extTop = tuple(c[c[:, :, 1].argmin()][0]) # get extreme top poin in mask
        # M = cv2.moments(mask)
        # cX = int(M["m10"] / M["m00"])
        
        # cY = int(M["m01"] / M["m00"])
        # # draw the contour and center of the shape on the image
        # # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        # mask[cY+50:bottomy,topx:bottomx] = 0.0
        # cv2.circle(mask, extTop, 7, (0, 0, 0), -1)
        # cv2.circle(mask, (cX, cY), 7, (0, 0, 0), -1)
        # cv2.putText(mask, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        

        # mask = mask[y:max_height, start_x:max_width]

        return mask
        # kebawah y semakin bertambah
    def masking_finger(self,img,mask):

        bottomx,bottomy,topx,topy = self.get_points(mask)
        
        
        # extTop = tuple(c[c[:, :, 1].argmin()][0]) # get extreme top poin in mask
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        
        cY = int(M["m01"] / M["m00"])
        
        # draw the contour and center of the shape on the image
        # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        mask[cY+100:bottomy,topx:bottomx] = 0.0
        
        # ellipse = cv2.fitEllipse(mask)
        # cv2.ellipse(mask,ellipse,(0,255,0),2)


        # bottomx,bottomy,topx,topy = self.get_points(mask)
        # print(bottomy)
        # if bottomx > 15:
            
        #     bottomx = bottomx-15
        # if topx > 15:
        #     topx = topx-15
        # height =0.6*bottomy
        # print(height)
        
        # if bottomy-topy > 530:
        #     bottomy = height

        # print( bottomx)
        # print(bottomy)
        # print(topy)
        # print(topx)
        mask_temp = mask.copy()

        mask_temp = mask_temp[int(topy):int(cY+100), int(topx):int(bottomx)]
        x,y,w,h = cv2.boundingRect(mask_temp)
        # print(w)
        if w>=320: # thumb
            mask = mask[int(topy):int(cY+100), int(topx):int(bottomx)]
        
            img = img[int(topy):int(cY+100), int(topx):int(bottomx)]
            
        else: #not thumb
            mask = mask[int(topy):int(cY-110), int(topx):int(bottomx)]
        
            img = img[int(topy):int(cY-110), int(topx):int(bottomx)]

        mask_result = cv2.bitwise_and(img, img, mask=mask)
        # mask_result = self.to_gray(mask_result)
        # x,y,w,h = cv2.boundingRect(mask_result)
        # x = x-15
        # w = w-15
        
        # mask_result = mask_result[y:y+h, x:x+w]

        
       
        return mask_result
    
    def smoothing(self,img):
        return cv2.GaussianBlur(img,(5,5),0)

    def normalize(self,img):

        image = np.copy(img)
        image -= np.min(image)
        m = np.max(image)
        if m > 0.0:
            image = image * 1.0 /m
          
        return image

    def localNormalize(self,image, w=SIZE):
        image = np.copy(image)
        height, width = image.shape
        for y in range(0, height, w):
            for x in range(0, width, w):
                image[y:y+w, x:x+w] = self.normalize(image[y:y+w, x:x+w])

        return image
    
    def resize_image(self,img,scale_percent=60):
        # scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
    def evaluate_image(self,img_path,img_ref,img_es,img_fr,img_e):
        esgran_metrics = ImageQualityMetrics()
        esgran_metrics_scores = esgran_metrics.compare_images(img_es,img_ref)
        #print('ESGRAN {}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(img_path,  esgran_metrics_scores[0],  esgran_metrics_scores[1],  esgran_metrics_scores[2]))

        fsrcnn_metrics = ImageQualityMetrics()
        fsrcnn_metrics_scores=fsrcnn_metrics.compare_images(img_fr,img_ref)

        # print all three scores with new line characters (\n) 
       # print('FSRCNN {}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(img_path, fsrcnn_metrics_scores[0], fsrcnn_metrics_scores[1], fsrcnn_metrics_scores[2]))

        edsr_metrics = ImageQualityMetrics()
        edsr_metrics_scores = edsr_metrics.compare_images(img_e,img_ref)
       # print('EDSR {}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(img_path,  edsr_metrics_scores[0],  edsr_metrics_scores[1],  edsr_metrics_scores[2]))
        return fsrcnn_metrics_scores,edsr_metrics_scores,esgran_metrics_scores
    
    def super_resolution(self,img,img_path,base):
        img = self.resize_image(img,60)
        img_name = base+ "_resize_{}.png".format(self.count)
        cv2.imwrite(img_name, img)

      

       

        sr_img = img.copy()
        sr2_img = img.copy()
        sr3_img = img.copy()

        start_time_f = time.time()
        img_fr = self.fsrcnn_sr.upsample_image(sr_img)
        end_f = time.time()
        # img_fr =self.gamma_correction(img_fr)
       
        
        img_name = base + "_fsrcnn_sr_{}.png".format(self.count)
        cv2.imwrite(img_name, img_fr)


        
        start_time_e = time.time()
        img_e = self.edsr_sr.upsample_image(sr2_img)
        end_e = time.time()
        # img_e = self.gamma_correction(img_e)
        
       
       
        img_name = base + "_edsr_sr_{}.png".format(self.count)
        cv2.imwrite(img_name, img_e)

        start_time = time.time()
        img_es = self.esgran_sr.upsample_image(sr3_img)
        end_time = time.time()
        # img_es= self.gamma_correction(img_es)
       
        
        img_name = base + "_esgran_sr_{}.png".format(self.count)
        cv2.imwrite(img_name, img_es)

        
        
        
        
        
        fsrcnn,edsr,esgran = self.evaluate_image(img_path,img,img_es,img_fr,img_e)

        res ={'file' : base,
                        'fsrcnn_time':end_f-start_time_f,
                        'edsr_time':end_e-start_time_e,
                        'esgran_time':end_time-start_time,
                        'fsrcnn_PSNR':fsrcnn[0],
                        'fsrcnn_MSE':fsrcnn[1],
                        'fsrcnn_SSIM':fsrcnn[2],
                        'edsr_PSNR':edsr[0],
                        'edsr_MSE':edsr[1],
                        'edsr_SSIM':edsr[2],
                        'esgran_PSNR':esgran[0],
                        'esgran_MSE':esgran[1],
                        'esgran_SSIM':esgran[2]}
                        
        # img_fr = self.to_gray(img_fr)
        # img_e = self.to_gray(img_e)
        # img_es = self.to_gray(img_es)

        # img_fr = self.normalize_image(img_fr)
        # img_e = self.normalize_image(img_e)
        # img_es = self.normalize_image(img_es)

        

        


        img_fr = self.smoothing(img_fr)
        img_e = self.smoothing(img_e) 
        img_es = self.smoothing(img_es) 

        # # img_es = self.smoothing(img_es)
        img_fr = self.automatic_brightness_and_contrast(img_fr)
        img_e = self.automatic_brightness_and_contrast(img_e)
        img_es = self.automatic_brightness_and_contrast(img_es)

        img_fr = self.increase_contrast(img_fr)
        img_e = self.increase_contrast(img_e)
        img_es = self.increase_contrast(img_es)
        
        
        


       
        
        
        
        img_es = self.adaptive_threshold(img_es)
        
        
        
        img_name = base + "_esgran_sr{}.png".format(self.count)
        filename = os.path.join(TEMP_PATH,img_name)
        cv2.imwrite(filename, img_es)
        cv2.imwrite(img_name, img_es)

        # img_fr,maxima,minima = self.detect_ridges(img_fr,3)
        img_fr = self.adaptive_threshold(img_fr)
        img_name = base + "_fsrcnn_sr_{}.png".format(self.count)
        filename = os.path.join(TEMP_PATH,img_name)
        cv2.imwrite(filename, img_fr)
        cv2.imwrite(img_name, img_fr)

       
        
        
      
        
        
        img_e = self.adaptive_threshold(img_e)
        img_name = base+ "_edsr_sr_{}.png".format(self.count)
        filename = os.path.join(TEMP_PATH,img_name)
        cv2.imwrite(filename, img_e)
        cv2.imwrite(img_name, img_e)

        return res


    # preprosess all the image:


    def preprocess_image(self,img_path,base):
        print(base)
        res={}
        img = cv2.imread(img_path)
        # img = cv2.flip(img,1)
        
        height,width,c = img.shape
        clipLimit=4
        size=(16,16)

        if height >= 1920 and width >= 1080:
            max_h = int(0.6*height)
            middle = width // 2

            img = img[1:max_h, middle-350:width]
        


            img_copy = img.copy()
       

            finger_masking = self.segment_finger(img_copy)
            
            

            
            # finger_masking = self.to_gray(finger_masking)
        
        
            

            img = self.masking_finger(img,finger_masking)
            # img_name = img_path + "_masked.png"
            # cv2.imwrite(img_name, img)
            

            # return

            # clipLimit=3
            # size=(16,16)
        # else:
        #     return
        
        # img_name = img_path + "_maskedresult_{}.png".format(self.count)
        # cv2.imwrite(img_name, img)
        img = self.to_gray(img)
        img_ref = img.copy()

        res = self.super_resolution(img_ref,img_path,base)
        
        
        
        # self.gamma_correction(img)
        img = self.normalize_image(img)
        img = self.smoothing(img)

        
        
        
        img = self.increase_contrast(img,clipLimit=clipLimit,size=size)
        
        


        
        
        



        
        
        # img = self.gamma_correction(img)
        #sblm dan sesudah smooth hasilnya beda, igt buat perbedaan,tambahan step
        


        
        
        img_x = img.copy()
        # img_x,maxima,minima = self.detect_ridges(img_x,3)
        
        # img = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=11)
        # img_name = "_sobel_ic_{}.png".format(self.count)
        # filename = os.path.join(TEMP_PATH,img_name)
        # cv2.imwrite(filename, img)
        # cv2.imwrite(img_name,img)
        

        # img_x = fingerprint_enhancer.enhance_Fingerprint(img_x)
        # kernel = np.ones((5,5),np.uint8)
        # img_x = cv2.morphologyEx(img_x, cv2.MORPH_CLOSE, kernel)


       
        img_x = self.adaptive_threshold(img_x)
        img_name = base + "_thres_{}.png".format(self.count)
        
        filename = os.path.join(TEMP_PATH,img_name)
        cv2.imwrite(filename, img_x)
        cv2.imwrite(img_name,img_x)


        
        
        

       
        self.count+=1
    
        return res

def get_matcher(image):
    url = 'http://localhost:8080/match'
    user_inbound = {'pathimage': image,'name':""}
    
    
    try:
        request = requests.get(url, json = user_inbound)
        result = request.json()
        info =['Nama : ' + result['data']['name'], 'Hasil penilaian : {:.2f}'.format(result['data']['score'])]
        
        return result
    except Exception as e: 
        print(e)
        pass
if __name__ == "__main__":
    
    p = PreprocessImage()
    # test_img_folder= "try_thumb/img_16.png"
    # p.preprocess_image(test_img_folder,"img_16")
    # test_img_folder = args[0]
    test_img_folder= "try_index/*"
    
    # idx=0

    

    res =pd.DataFrame(columns=['file', 'fsrcnn_time','edsr_time','esgran_time','fsrcnn_PSNR','fsrcnn_MSE','fsrcnn_SSIM','edsr_PSNR','edsr_MSE','edsr_SSIM','esgran_PSNR','esgran_MSE','esgran_SSIM'])
    for path in glob.glob(test_img_folder):
        
        base = osp.splitext(osp.basename(path))[0]
   
       
        start_time = time.time()
        preproses_res = p.preprocess_image(path,base)
        # res = res.append(preproses_res,ignore_index=True)
        print("total execution : ---  %s seconds ---" % (time.time() - start_time))
        
    res.to_excel('./sr.xlsx',sheet_name='SR')

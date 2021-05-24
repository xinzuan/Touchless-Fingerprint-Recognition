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

TEMP_PATH ='/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/backend/complete/src/resources/temp/'

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

    	
    test_img_folder=TEMP_PATH+"*"
    res =pd.DataFrame(columns=['file', 'Name','Score','Match_Time'])
    excel_file ='./match-mix.xlsx'
    
    for path in glob.glob(test_img_folder):
        
        base = osp.splitext(osp.basename(path))[0]
        start_time = time.time()
        result = get_matcher(path)
        
        execution = time.time()
        print("total execution : ---  %s seconds ---" % (time.time() - start_time))
        # if result['data']['score'] >=40:
        print(base)
        
           
        res =res.append({'file' : base,
                    'Name' : result['data']['name'], 
                    'Score': result['data']['score'],
                    'Match_Time':execution-start_time},
                    ignore_index=True)
        
    res.to_excel(excel_file,sheet_name='Matching')
     

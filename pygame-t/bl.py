
import cv2

import numpy as np

import os.path as osp
import glob

from preprocess import PreprocessImage

# 11-12
# 16-19
# Write some Text

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2



def check_blurry(path,img,b,threshold_min=9, threshold_max=12):
    blur_value = cv2.Laplacian(img, cv2.CV_64F).var()
    # if blur_value > 0 :
    #     cv2.putText(img, str(blur_value), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))
    # # print(base,blur_value)
    #     cv2.imwrite(path, img)
    # is_blur =  blur_value > threshold_min and  blur_value < threshold_max
    # # print()
    return blur_value

test_img_folder = 'img/*'



idx = 0

for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    # print(idx, base)
    # read images
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    c = check_blurry(path,img,base)

    print (base,c)
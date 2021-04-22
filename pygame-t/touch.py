import cv2
import numpy as np

import os.path as osp
import glob
def process(img_path):
    img = cv2.imread(img_path,0)
            
    # height,width,c = img.shape

    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    dim = (500, 764)
        
        # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img_name = img_path + "_binary.png"

    cv2.imwrite(img_name, img)

test_img_folder = 'ref/*'
idx=0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(base)
    # read images

    
    
    process(path)
    
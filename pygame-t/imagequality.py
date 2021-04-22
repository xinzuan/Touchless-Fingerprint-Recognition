from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import math

# psnr higher better
# ssim = 1 -> perfect
class ImageQualityMetrics:
    def __init__(self):
        self.scores = []
    def psnr(self,target, ref):

        target_data = target.astype(float)
        ref_data = ref.astype(float)

        diff = ref_data - target_data
        diff = diff.flatten('C')

        rmse = math.sqrt(np.mean(diff ** 2.))

        return 20 * math.log10(255. / rmse)

# define function for mean squared error (MSE)
    def mse(self,target, ref):
        # the MSE between the two images is the sum of the squared difference between the two images
        err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
        err /= float(target.shape[0] * target.shape[1])
        
        return err

    # define function that combines all three image quality metrics
    def compare_images(self,target, ref):
       
        if target.shape != ref.shape:
            if ref.ndim == 2:

                height,width = ref.shape

            if ref.ndim == 3:
                height,width,_ = ref.shape

    
            
            
            dim = (width, height)
            
            # resize image
            target = cv2.resize(target, dim, interpolation = cv2.INTER_AREA)
        
        self.scores.append(self.psnr(target, ref))
        self.scores.append(self.mse(target, ref))
        self.scores.append(ssim(target, ref, multichannel =True))
        
        return self.scores
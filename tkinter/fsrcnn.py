import cv2
from cv2 import dnn_superres
import time
# edsr around 5 minutes
# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('./img_0.jpg')

# Read the desired model
print("read model...")
path = "FSRCNN_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
print("set model...")
sr.setModel("fsrcnn", 4)

# Upscale the image
print("upsacling..")
start = int(round(time.time() * 1000))
result = sr.upsample(image)
stop = int(round(time.time() * 1000))

# Save the image

cv2.imwrite("./upa.jpg", result)
print("saved")
print("time : ",stop-start)

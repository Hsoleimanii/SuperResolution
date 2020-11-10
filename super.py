import cv2
from cv2 import dnn_superres

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('2.jpg')

# ##########Read the desired model
#path = "./models/EDSR_x3.pb"
path = "./models/LapSRN_x2.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 3)

# Upscale the image
result = sr.upsample(image)



cv2.imshow("Original Image", image)
cv2.imshow("Super Resolution by bicubic", cv2.resize(image,None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
cv2.imshow("Super Resolution by DL", result)
key = cv2.waitKey(20000) 
cv2.destroyAllWindows()

# Save the image
cv2.imwrite("./upscaled.png", result)


#OK
###############################################   if you want to use GPU
# Read the desired model
"""
path = "EDSR_x3.pb"
sr.readModel(path)

# Set CUDA backend and target to enable GPU inference
sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
"""
# Standard imports
import cv2
import numpy as np;
 
# Read image
img = cv2.imread("container.jpg", cv2.IMREAD_GRAYSCALE)
_ , img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)
img = cv2.medianBlur(img, 5)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

params.filterByColor = False
 
# Change thresholds
params.minThreshold = 0;
# params.maxThreshold = 00;
 
# Filter by Area.
params.filterByArea = False
params.minArea = 300
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.25
params.maxCircularity = 0.87
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.25
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
params.maxInertiaRatio = 0.5
 

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(img)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", img_with_keypoints)
cv2.waitKey(0)
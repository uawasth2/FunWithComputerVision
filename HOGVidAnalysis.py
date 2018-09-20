import cv2
import numpy as np
import glob
from PIL import Image

# Set up SVM for OpenCV 3
svm = cv2.ml.SVM_load('digits_svm_model.yml')

# these are the stuff that are required to make a hog function
# size of image
winSize = (600,800)
#size of area you will check at a time
blockSize = (150,200)
# how much your blocks will shift each iteration
blockStride = (75,100)
# the number of cells that blocks will be split into for work
cellSize = (75,100)
# the number of histogram bins
nbins = 9
# the below dont ever change
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
# whether you want your vectors to be able to be negative
useSignedGradients = False
 
# create your hog model
hog = cv2.HOGDescriptor(winSize,blockSize,
    blockStride,cellSize,nbins,
    derivAperture,winSigma,histogramNormType
    ,L2HysThreshold,gammaCorrection,nlevels,
    useSignedGradients)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break   

    for i in range(0, len(frame) - 120, 120):
        for j in range(0, len(frame[i]) - 160, 160):
            img = Image.fromarray(frame[i:i+120, j:j+160])
            img = img.resize((600, 800))
            img = np.array(img)
            testData = []
            descriptor = hog.compute(img)
            testData.append(descriptor)
            testData = np.array(testData)
            predictions = svm.predict(testData)[1].ravel()
            if predictions != 0:
                cv2.rectangle(frame, (j, i), (j + 160, i + 120), 255, 5)
    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        break
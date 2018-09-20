import cv2
import numpy as np
import glob
from PIL import Image

# finds accuracy of your model
# takes in model, test images, test hog data, and test labels
def svmEvaluate(model, digits, samples, labels):

    # Test on a held out test set
    predictions = svm.predict(testData)[1].ravel()
    print(predictions)
    accuracy = (labels == predictions).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy*100))

    confusion = np.zeros((2, 2), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

# gets all image file names from a folder
neg = glob.glob('./neg/*')
pos = glob.glob('./pos/*')

# where training and test images and their labels will be located
trainingImages = []
trainingLabels = []
testImages = []
testLabels = []

# loads the images from their files and appends them to the arrays
# image.load closes the file pathing and saves memroy
for img in neg[0:-50]:
    image = Image.open(img)
    trainingImages.append(image)
    image.load()
    trainingLabels.append(0)


for img in pos[0:-50]:
    image = Image.open(img)
    trainingImages.append(image)
    image.load()
    trainingLabels.append(1)

for img in neg[-50:]:
    image = Image.open(img)
    testImages.append(image)
    image.load()
    testLabels.append(0)
for img in pos[-50:]:
    image = Image.open(img)
    testImages.append(image)
    image.load()
    testLabels.append(1)



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

trainingData = []
testData = []

# take your images and resize them into a standard size
# then turn them into a np array
# then comput the hog vector and append it to training data
for img in trainingImages:
    img = img.resize((600, 800))
    img = np.array(img)
    trainingData.append(hog.compute(img))

for img in testImages:
    img = img.resize((600, 800))
    img = np.array(img)
    descriptor = hog.compute(img)
    testData.append(descriptor)

# convert data to arrays
trainingData = np.array(trainingData, dtype = np.float32)
testData = np.array(testData)

# Set up SVM for OpenCV 3
svm = cv2.ml.SVM_create()
# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)
# Set SVM Kernel to Radial Basis Function (RBF) 
svm.setKernel(cv2.ml.SVM_RBF)
# Set parameter C
svm.setC(12.5)
# Set parameter Gamma
svm.setGamma(0.50625)

# Train SVM on training data  
svm.trainAuto(np.array(trainingData), cv2.ml.ROW_SAMPLE, np.array(trainingLabels))
 
# Save trained model 
svm.save("digits_svm_model.yml")

# test how well hog worked for you
svmEvaluate(svm, testImages, testData, testLabels)
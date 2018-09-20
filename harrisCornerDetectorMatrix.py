import cv2
import numpy as np

# returns quadratic solutions
def quadratic(a, b, c):
    d = -1 * b + np.sqrt(b ** 2 - 4 * a * c )
    e = -1 * b - np.sqrt(b ** 2 - 4 * a * c )
    d = d / (2 * a)
    e = e / (2 * a)
    return d, e

# returns eigenvalues for a 2x2 matrix
def eigenvalue_finder(matrix):
    a = 1
    b = -1 * (matrix[0][0] + matrix[1][1])
    c = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return quadratic(a, b, c)

# goals 
# test small areas, get their x and y changes
# get the matrix and find eigenvalues
# u and v do not matter
# try and find regions where the R = det(M) - k (trace(M))^2 is low
# k is between 0.04 and 0.06
# R is large for corner
# R is negative with large magnitude for edge
# abs(R) is small for a flat region


# returns the delta change in x and y direction for a pixel
def delt_vals_of_image(img):
    # use a 3x3 pixel region
    # region = [3, 3]

    # the array of delta values for entire image
    delta_vals = []
    
    # looks at all non border pixels
    for x in range(1, len(img) - 1):
        # separates values by row, same shape as picture
        arr = []
        for y in range(1, len(img[x]) - 1):
            # captures change in pixel values by ignoring the current pixel itself and instead looks at pixels directly above, below, right, left
            delta_x = int(img[x + 1][y]) - int(img[x - 1][y])
            delta_y = int(img[x][y + 1]) - int(img[x][y - 1])
            arr.append([delta_x, delta_y])
        delta_vals.append(arr)
    return delta_vals

# gets the r values of each pixel in an image
def regional_r_vals(img):
    # region that is examined and summed
    region = [5, 5]
    # shift between each window
    shift = 5
    # converts input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gets the x,y change for each pixel in image
    delts = delt_vals_of_image(gray)
    # where we will store all our r values 
    r_val_array = []
    # goes through designated windows with shift between them
    for x in range(0, len(delts) - region[0], shift):
        # to be in the same shape as the image
        arr = []
        for y in range(0, len(delts[x]) - region[1], shift):
            # summation of Ix^2, Iy^2 and Ix*Iy in your window range
            delt_xsquared_sum = 0.0
            delt_ysquared_sum = 0.0
            delt_xy_sum = 0.0
            for i in range(x, x + region[0]):
                for j in range(y, y + region[1]):
                    delt_xsquared_sum += delts[i][j][0] ** 2
                    delt_ysquared_sum += delts[i][j][1] ** 2
                    delt_xy_sum += delts[i][j][0] * delts[i][j][1]
            
            # creates a finalized matrix with the sums, finds eigenvalues and then r values
            matrix = [[delt_xsquared_sum, delt_xy_sum],[delt_xy_sum, delt_ysquared_sum]]
            l1, l2 = eigenvalue_finder(matrix)
            determinant = l1 * l2
            trace = l1 + l2
            # The Shi-Tomasu system would use the min of the eigenvalues
            # r_val = min(l1, l2)
            r_val = determinant - 0.05 * (trace ** 2) 
            # adds x,y location of pixel and r value
            arr.append([x, y, r_val])
        r_val_array.append(arr)
    return r_val_array

# overall function for corner detection
def corner_detector(img, a):
    # gets r values
    r_array = regional_r_vals(img)
    # goes through all r values in array
    for i in r_array:
        for j in i:
            # if the r value is high enough to be considered a corner, draw a rectangle around it
            # Shi-Tomasu would use a much lower threshold
            if j[2] > 30000:
                cv2.rectangle(img, (j[1], j[0]), (j[1] + a, j[0] + a), 255, 1)
                
image = cv2.imread("../OpenCVTutorials/opencv-corner-detection-sample.jpg")
corner_detector(image, 5)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

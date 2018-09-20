from __future__ import division
import numpy as np
import cv2

# gray scaled image
image = cv2.imread("../OpenCVTutorials/opencv-corner-detection-sample.jpg")
# displacement to use
uv_example = [2, 2]
# window size to sum and use as window function
# first two are pix location and second two are size of window
window_size_example = [250, 150, 100, 100]


# calculates the shift value for a certain window, params are: image, window that we want, and uv is x/y shift
def shift_calculator(img, winsize, uv):

    # the sum
    sigma = 0.0

    # the x and y shifts respectively
    u = uv[0]
    v = uv[1]

    # goes through each pixel in region/window
    for x in range(winsize[0], winsize[0] + winsize[2], 10):

        # checks if x or x + u goes out of range
        if x < 0 or x >= len(img) or x + u < 0 or x + u >= len(img):
            continue
        
        for y in range(winsize[1], winsize[1] + winsize[3], 10):
            
            # checks if y or y + v goes out of range
            if y < 0 or y >= len(img[0]) or y + v < 0 or y + v >= len(img[0]):
                continue

            # sums the change/contrast of the original pixel and the pixel at shifted location
            sigma += (int(img[x + u, y + v]) - int(img[x, y]))**2

        # divides by 10 so no overflow?    
        sigma = sigma / 10
    # divides by 10 again
    sigma = sigma / 10

    # returns the summed value at the location
    return sigma


# method that gets all possible values of shifts in a region and then combines them into an array with the x-center value, y-center value, and then the score
def corner_detector(img):
    # manually determined shift and window size for a square window
    shift = 10
    region = 10

    # array to be filled with x,y location and then the value at location
    final_img_values = []

    # gets length and height of image
    w = len(img)
    h = len(img[0])

    # goes through many regions in imagewith tiny shifts horizontally
    for x in range(0, w, 3):

        # used to partition the overall array by rows
        arr = []

        # goes through many regions in image with tiny shifts vertically
        for y in range(0, h, 3):

            # using an array and a sum function to add shifts in all 8 cardinal compass directions using shift_calculator
            box_score = []
            box_score.append(shift_calculator(img, [x, y, region, region], [shift, 0]))
            box_score.append(shift_calculator(img, [x, y, region, region], [shift, shift]))
            box_score.append(shift_calculator(img, [x, y, region, region], [0, shift]))
            box_score.append(shift_calculator(img, [x, y, region, region], [-shift, shift]))
            box_score.append(shift_calculator(img, [x, y, region, region], [-shift, 0]))
            box_score.append(shift_calculator(img, [x, y, region, region], [-shift, -shift]))
            box_score.append(shift_calculator(img, [x, y, region, region], [0, -shift]))
            box_score.append(shift_calculator(img, [x, y, region, region], [shift, -shift]))
            total = sum(box_score)

        # appending the arrays to get it in array split by row and then col    
            arr.append([x, y, total])
        final_img_values.append(arr)

    return final_img_values

# function that calculates the regional maxima in our x,y,value array called scores
def maxima_only(scores):

    # creates a copy of the scores array
    maxima = list(scores)

    # goes through our entire array
    for i in range(len(scores)):
        for j in range(len(scores[i])):

            # value to go to next iteration if this iteration's work is done
            escape = False

            # m is the range of maxima that you may want to look around
            m = 10

            # goes through a range, m, of all our values around the current pixel
            for k in range(i - m, i + m + 1):
                for l in range(j - m, j + m + 1):

                    # out of bounds check
                    if k < 0 or l < 0 or k >= len(scores) or l >= len(scores[i]):
                        continue

                    # if it is less than something around it, it is not a maxima    
                    if scores[i][j][2] <= scores[k][l][2]:

                        # makes sure that it doesn't eliminate a maxima because it is equal to itself
                        if i == k and j == l:
                            continue

                        # if not a maxima void out the value and try to break current regional check    
                        maxima[i][j][2] = 0
                        escape = True
                        break

                if escape:
                    break

            if escape:
                continue
    return maxima

            


def corner_box_drawer(img):
    # convert our image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect all possible corners in our image
    scores = corner_detector(gray)
    # eliminate most edges and get only ones with highest values
    scores = maxima_only(scores)

    # draw an rectangel around all the values which have been detected as maxima
    for i in scores:
        for j in i:
            if j[2] > 0:
                cv2.rectangle(img, (j[1] - 3, j[0] - 3), (j[1] + 3, j[0] + 3), 0, 1)
    
# calls all our functions
corner_box_drawer(image)
# show image
cv2.imshow('image', image)
# wait for any key before exiting
cv2.waitKey()
# destroy all active windows when exiting
cv2.destroyAllWindows()
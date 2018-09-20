import cv2
import numpy as np

a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = np.subtract(a, b)
print(c)

# applies a kernel and modifies image through cross filtering
def kernel_filtering(img, kernel):
    # image to modify
    modified = list(img)
    np.array(modified, dtype = float)
    # the distance away from the edge of the kernel is the central pixel
    move = len(kernel) // 2
    # goes through each pixel in image that will not cause the kernel to go out of bounds
    for x in range(move, len(img) - move):
        for y in range(move, len(img[x]) - move):

            tot_sum = 0.0
            # once you get central pixel, will go through each pixel surrounding it
            for i in range( -1 * move, move + 1):
                for j in range( -1 * move, move + 1):
                    img_pix_val = float(img[x + i][y + j])
                    kernel_pix_val = float(kernel[move + i][move + j])
                    # sum up the pix_val weighted by the kernel weighting
                    tot_sum += img_pix_val * kernel_pix_val
            modified[x][y] = tot_sum
    return modified

def ghost(img):
    # image to modify
    modified = list(img)
    np.array(modified, dtype = float)
    move = 1
    # goes through each pixel in image that will not cause the kernel to go out of bounds
    for x in range(move, len(img) - move):
        for y in range(move, len(img[x]) - move):
            equal = 0
            eq_sum = 0.0
            darker = 0
            dark_sum = 0.0
            lighter = 0
            light_sum = 0.0
            # once you get central pixel, will go through each pixel surrounding it
            for i in range( -1 * move, move + 1):
                for j in range( -1 * move, move + 1):
                    if i == 0 and j == 0:
                        continue
                    if img[x][y] > img[x + i][y + j]:
                        lighter += 1
                        light_sum += img[x + i][y + j]
                    elif img[x][y] < img[x + i][y + j]:
                        darker += 1
                        dark_sum += img[x + i][y + j]
                    else:
                        equal += 1
                        eq_sum += img[x + i][y + j]
            if darker == lighter:
                if darker > equal:
                    modified[x][y] = (dark_sum + light_sum) / (darker + lighter)
                else:
                    modified[x][y] = eq_sum / equal          
            elif darker == equal:
                if darker > lighter:
                    modified[x][y] = (dark_sum) / (darker)
                else:
                    modified[x][y] = light_sum / lighter
            elif lighter == equal:
                if lighter > darker:
                    modified[x][y] = (light_sum) / (lighter)
                else:
                    modified[x][y] = dark_sum / darker
            elif darker > lighter and darker > equal:
                modified[x][y] = dark_sum / darker
            elif lighter > darker and lighter > equal:
                modified[x][y] = light_sum / lighter
            elif equal > darker and equal > lighter:
                modified[x][y] = eq_sum / equal
    return modified


# take in original pic and display
einstein = cv2.imread('einstein-albert-head-raw.jpg', 0)
cv2.imshow('original', einstein)

# # create a smoothing kernel
# kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
# kernel = np.array(kernel)
# kernel.astype(float)
# kernel = kernel / float(9)

# # create a sharpening filter
# kernel2 = [[0,0,0],[0,2,0],[0,0,0]]
# kernel2 = np.array(kernel2, dtype = float)
# kernel2 = np.subtract(kernel2, kernel) 
# print(kernel2)

# # display a smoothing filter
# einstein_smoothed = kernel_filtering(einstein, kernel)
# einstein_smoothed = np.array(einstein_smoothed)
# cv2.imshow('smoothed', einstein_smoothed)

# # display a sharpening filter
# einstein_sharpened = np.array(kernel_filtering(einstein, kernel2))
# cv2.imshow('details', einstein_sharpened)

# ghoster
einsten_ss = np.array(ghost(einstein))
cv2.imshow('ghost homemade', einsten_ss)

cv2.waitKey()
import numpy as np

original_matrix = [[46,24,13,65],[64,82,32,87],[12,48,34,75],
    [61,64,32,74],[5,28,18,34]]

def row_mean(matrix, num):
    row = matrix[num]
    sigma = sum(row)
    return sigma / len(row)

def col_mean(matrix, num):
    transposed = np.transpose(matrix)
    col = transposed[num]
    sigma = sum(col)
    return sigma / len(col)

def covariance_of_matrix(matrix):
    # covariance matrix being made
    covariance = []
    #for each column in matrix
    for j in range(len(matrix[0])):
        arr = []
        #for each column in matrix
        for k in range(len(matrix[0])):
            #num rows in matrix
            size = len(matrix)
            #summation of covariance function
            sigma = 0
            # for each value in a column
            for i in range(len(matrix)):
                # gets difference of value in matrix and the mean of the row it is in
                j_diff = matrix[i][j] - col_mean(matrix, j) 
                k_diff = matrix[i][k] - col_mean(matrix, k)
                # adds the summation of the product of the two values
                sigma += j_diff * k_diff
            # divide by the size 
            arr.append(sigma/size)
        covariance.append(arr)
    return covariance

print(covariance_of_matrix(original_matrix))
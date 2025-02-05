#!/usr/bin/env python3
import sys
import numpy as np

def get_quant_error(matrix, quantNum):
    quantmult = np.float32(127)/np.float32(quantNum)

    #Quantize
    quantized_matrix = np.around((matrix*quantmult)).astype(np.int8)

    #Unquantize
    unquantized_matrix = quantized_matrix*(1/quantmult)

    return np.square(np.subtract(unquantized_matrix, matrix)).mean()

def find_best_matrix(matrix):
    maxAbs = np.float32(abs(max(matrix.min(), matrix.max(), key=abs)))
    maxAbs = maxAbs + np.float32(0.001) # The first condition
    bestFactor = maxAbs
    bestMSE =  np.inf
    limit = 0.03*maxAbs

    while maxAbs > limit:
        maxAbs = maxAbs - np.float32(0.001) # The first condition
        mse = get_quant_error(matrix, maxAbs)
        #print("MSE", mse)
        if mse < bestMSE:
            bestMSE = mse
            bestFactor = maxAbs


    # Now that we found the best matrix, replace the Maximum (or minimum number) in the matrix with that number
    # so that it can be nicely picked up by our marian implementation

    maxNum = abs(max(matrix.min(), matrix.max(), key=abs))
    minNum = -maxNum
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > bestFactor:
                matrix[i][j] = bestFactor
            elif matrix[i][j] < -bestFactor:
                matrix[i][j] = -bestFactor

    print("Old MaxAbs:", maxNum, "new:", bestFactor, "actual:", abs(max(matrix.min(), matrix.max(), key=abs)))


    return matrix

def find_best_matrix_kenneth(matrix):
    maxAbs = np.float32(abs(max(matrix.min(), matrix.max(), key=abs)))
    min_norm = np.inf
    bestFactor = maxAbs

    for i in range(1,300):
        m = maxAbs * float(i) / float(300)
        norm = np.linalg.norm(np.clip(np.around(matrix * 127. / m), -127., 127.) / (127. / m) - matrix)
        if norm < min_norm:
            bestFactor = m
            min_norm = norm

    maxNum = abs(max(matrix.min(), matrix.max(), key=abs))
    minNum = -maxNum
#    print("Old MaxAbs:", maxNum, "new:", bestFactor)

#    matrix = np.where(matrix >= maxNum, bestFactor, matrix)
#    matrix = np.where(matrix <= minNum, -bestFactor, matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > bestFactor:
                matrix[i][j] = bestFactor
            elif matrix[i][j] < -bestFactor:
                matrix[i][j] = -bestFactor
    print("Old MaxAbs:", maxNum, "new:", bestFactor, "actual:", abs(max(matrix.min(), matrix.max(), key=abs)))
    return matrix

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage:", sys.argv[0], "input_model output_model")
        sys.exit(1)

    model_file = np.load(sys.argv[1])
    model_file_dict = dict(model_file)

    for matrix in model_file_dict.keys():
#        if matrix[-2] == 'W' or matrix == "Wemb":
#            model_file_dict[matrix] = (model_file_dict[matrix]*.9).astype(np.float32)

        if matrix[-2] == 'W' or matrix == "Wemb":
            print(matrix)
            model_file_dict[matrix] = find_best_matrix_kenneth(model_file_dict[matrix])
#        elif matrix == "Wemb":
#            model_file_dict[matrix] = (model_file_dict[matrix]*.9).astype(np.float32)


    # Save
    np.savez(sys.argv[2], **model_file_dict)

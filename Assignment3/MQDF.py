# coding=utf8

import math
import numpy as np
from numpy import linalg
import csv
import timeit



def load_data(scaled=False):

    print('Loading dataset...')
    data_class_list = set()
    train_x = []
    train_y = []
    
    with open('mnist_train.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row == []:
                continue
            train_y.append(row[-1])
            data_class_list.add(int(row[-1]))
            train_x.append(row[:-1])

    test_x = []
    test_y = []
    with open('mnist_test.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row == []:
                continue
            test_y.append(row[-1])
            data_class_list.add(int(row[-1]))
            test_x.append(row[:-1])

    class_num = len(data_class_list)
    feature_num = len(train_x[0])

    train_x = np.array(train_x, np.float64)
    train_y = np.array(train_y, np.int)
    test_x = np.array(test_x, np.float64)
    test_y = np.array(test_y, np.int)

    if scaled:
        train_length = len(train_x)
        x_whole = np.vstack((train_x, test_x))
        from sklearn.preprocessing import scale
        x_whole = scale(x_whole, axis=0, with_std=True)

        train_x = x_whole[:train_length, :]
        test_x = x_whole[train_length:, :]

    print("Load data end!")
    return class_num, feature_num, train_x, train_y, test_x, test_y

def MQDF(class_num, train_x, train_y, k):
    """
    Build the MQDF model
    """

    feature_num_ = len(train_x[0])   # number of features
    assert(k<feature_num_ and k>0)

    data = []
    train_length = len(train_x)
    for i in range(class_num):
        data.append(list())

    for i in range(train_length):
        class_index = int(train_y[i])
        data[class_index].append(train_x[i])

    mean = []
    cov_matrix = []
    prior = []

    for i in range(class_num):
        data[i] = np.matrix(data[i], dtype=np.float64)
        mean.append(data[i].mean(0).T)

        # np.cov treat each row as one feature, so data[i].T has to be transposed
        cov_matrix.append(np.matrix(np.cov(data[i].T)))
        prior.append(len(data[i]) * 1.0 / train_length)

    
    eigenvalue_list = []    # store the first largest k eigenvalue_lists of each class
    eigenvector_list = []   # the first largest k  eigenvector_lists, column-wise of each class
    delta = [0] * class_num # delta for each class
    for i in range(class_num):
        covariance = cov_matrix[i]
        eig_value, eig_vector = linalg.eigh(covariance)

        # sort the eigvalues
        index_ = eig_value.argsort()
        index_ = index_[::-1]   # reverse the array
        eig_value = eig_value[index_]
        eig_vector = eig_vector[:, index_]
        
        eigenvector_list.append(eig_vector[:, 0:k])
        eigenvalue_list.append(eig_value[:k])
        
        # delta via ML estimation
        delta[i] = (covariance.trace() - sum(eigenvalue_list[i])) * 1.0 / (feature_num_ - k)
        
    return mean, eigenvalue_list,  eigenvector_list, delta
    
    
def MQDF_predict(test_x, class_num, k, mean, eigenvalue_list,  eigenvector_list, delta, test_y):
    """ MQDF predict
    """
    d = len(test_x[0])
    predict_lists = []
    testdata_num = 0

    
    for row in test_x:
        x = np.matrix(row, np.float64).T
        min_posteriori = float('inf')
        prediction = -1

        # formula
        for i in range(class_num):

            minus = np.linalg.norm(x.reshape((d,)) - mean[i].reshape((d,))) ** 2
            matrix_minus = [0] * d
            for j in range(k):
                matrix_minus[j] = (((x - mean[i]).T * eigenvector_list[i][:, j])[0,0])**2
            
            g = 0
            for j in range(k):
                g += (matrix_minus[j] * 1.0 / eigenvalue_list[i][j])
            
            g += ((minus - sum(matrix_minus)) / delta[i])
            
            for j in range(k):
                g += math.log(eigenvalue_list[i][j])
                
            g += ((d - k) * math.log(delta[i]))

                
            if g < min_posteriori:
                min_posteriori = g
                prediction = i
                
        predict_lists.append(prediction)

        testdata_num += 1

    return predict_lists


# load data
class_num, feature_num, train_x, train_y, test_x, test_y = load_data()

starttime = timeit.default_timer()

# K-L
k = 30
mean, eigenvalue_list, eigenvector_list, delta = MQDF(class_num, train_x, train_y, k)

# MQDF predict

predict_lists = MQDF_predict(test_x, class_num, k, mean, eigenvalue_list,  eigenvector_list, delta, test_y)

endtime = timeit.default_timer()
print("Code running: %.2fmin" % ((endtime - starttime) / 60.))

accuracy = np.mean((test_y == predict_lists)) * 100.0
print('Final correct:', accuracy)






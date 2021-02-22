# import the lib
import numpy as np
import pandas as pd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt


# read the dataser iris
data = pd.read_csv('iris.data',header=None)

# set the seed
random.seed(17)

# mapping dataset
data[4] = data[4].map({'Iris-setosa':0, 
                           'Iris-versicolor':1,
                           'Iris-virginica':2})

# KNN
class KNN:
    
    def __init__(self, k):
        
        """
        K is the neighbor number
        """
        self.k = k
    
    # Because KNN is an inert algorithm, no training process is required
    def train_KNN(self, X, y):
        """
        X:The first four columns of data used in training
        y:The tag column of the data used in the training
        """
        #  Convert X, y to type ndarray
        
        self.X = np.asarray(X)    
        self.y = np.asarray(y)   
    
    # to predict
    def Predict(self, X):         
        """
        According to the samples passed by parameters 
        the sample data is predicted
           
        Return：
        result：predict result
        """
        X = np.asarray(X)
        result = []
        
        # Iterate through the ndarray array, fetching one row at a time.
        
        
        for x in X:
            """
            For each sample in the test set, 
            the distance is calculated from the sample in the training set in turn
            """
            #Compute the Euclidean distance
            distance = np.sqrt(np.sum((x - self.X) ** 2, axis = 1))


            """
            After sorting the distance array, 
            returns the index of each element in the original array (the array before sorting)
            """
            index = distance.argsort()
            
            """
            Truncate the distance index array a
            take the index of the first K elements
            """            
            index = index[:self.k]


            """
            Count the number of occurrences of each element in the index array
            """
            count = np.bincount(self.y[index])


            """
            Returns the index corresponding to the element with the most occurrences
            which is the label corresponding to this set of data
            """
            result.append(count.argmax())
                        
            
        
        return np.asarray(result)


"""
process the dataset
"""
#Extract the data for each category 
t0 = data[data[4] == 0]
t1 = data[data[4] == 1]
t2 = data[data[4] == 2]

# Randomly scramble each set of data
t0 = t0.sample(len(t0), random_state = 17)
t1 = t1.sample(len(t1), random_state = 17)
t2 = t2.sample(len(t2), random_state = 17)

#Camouflage a train data
train_X = pd.concat([t0.iloc[:40, :-1], t1.iloc[:40, :-1], t2.iloc[:40, :-1]],axis=0)
train_y = pd.concat([t0.iloc[:40, -1], t1.iloc[:40, -1], t2.iloc[:40, -1]],axis=0)

# test data and use 5 cross validation
test_X_1 = pd.concat([t0.iloc[40: , :-1], t1.iloc[40: , :-1], t2.iloc[40: , :-1]],axis=0)
test_y_1 = pd.concat([t0.iloc[40: , -1], t1.iloc[40: , -1], t2.iloc[40: , -1]],axis=0)
test_X_2 = pd.concat([t0.iloc[0:10, :-1], t1.iloc[0:10 , :-1], t2.iloc[0:10 , :-1]],axis=0)
test_y_2 = pd.concat([t0.iloc[0:10 , -1], t1.iloc[0:10 , -1], t2.iloc[0:10 , -1]],axis=0)
test_X_3 = pd.concat([t0.iloc[10:20, :-1], t1.iloc[10:20 , :-1], t2.iloc[10:20 , :-1]],axis=0)
test_y_3 = pd.concat([t0.iloc[10:20 , -1], t1.iloc[10:20 , -1], t2.iloc[10:20 , -1]],axis=0)
test_X_4 = pd.concat([t0.iloc[20:30, :-1], t1.iloc[20:30 , :-1], t2.iloc[20:30 , :-1]],axis=0)
test_y_4 = pd.concat([t0.iloc[20:30 , -1], t1.iloc[20:30 , -1], t2.iloc[20:30 , -1]],axis=0)
test_X_5 = pd.concat([t0.iloc[30:40, :-1], t1.iloc[30:40 , :-1], t2.iloc[30:40 , :-1]],axis=0)
test_y_5 = pd.concat([t0.iloc[30:40 , -1], t1.iloc[30:40 , -1], t2.iloc[30:40 , -1]],axis=0)


result_re = []
for i in range(1,100):
    
    knn = KNN(k = i)
    # train
    knn.train_KNN(train_X, train_y)

    # test
    result_1 = knn.Predict(test_X_1)
    result_2 = knn.Predict(test_X_2)
    result_3 = knn.Predict(test_X_3)
    result_4 = knn.Predict(test_X_4)
    result_5 = knn.Predict(test_X_5)
    
    #Computational accuracy
    r_1 = np.sum(result_1 == test_y_1) / len(result_1)
    r_2 = np.sum(result_2 == test_y_2) / len(result_2)
    r_3 = np.sum(result_3 == test_y_3) / len(result_3)
    r_4 = np.sum(result_4 == test_y_4) / len(result_4)
    r_5 = np.sum(result_5 == test_y_5) / len(result_5)

    avg = (r_1 + r_2 + r_3 + r_4 + r_5) / 5
    result_re.append(avg)



# Draw a graph to show the change of accuracy with K value
plt.title('KNN with cross-fold validation')
plt.xlabel("h")
plt.ylabel("accuracy")
plt.plot(result_re)
plt.show






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

"""
process the dataset
"""
#Extract the data for each category 
t0 = data[data[4] == 0]
t1 = data[data[4] == 1]
t2 = data[data[4] == 2]

# let the data as array
data = data.values
t0 = np.mat(t0)
t1 = np.mat(t1)
t2 = np.mat(t2)

#random the data
np.random.shuffle(data)

# train data
train_0 = t0[0:30,:-1]
train_1 = t1[0:30,:-1]
train_2 = t2[0:30,:-1]

# verify data
verify = np.vstack((t0[30:40,],t1[30:40,],t2[30:40,]))
verify_X = verify[0:30,:-1]
verify_Y = verify[0:30,-1]


"""
calculate
"""

# calculate ∅
def get_phi(x, xi, h):
    phi = 0
    x = np.mat(x)
   
    xi = np.mat(xi)
    phi =  np.exp(-(x - xi) * (x - xi).T / (2 * h * h))

    return phi

# Calculate the overall formula
def get_px(x, xi, h):
    px = 0
    phi = 0

    for T_t in xi:
        phi += get_phi(x, T_t, h)
            
    px = phi  / ( len(xi) * np.power(h, 3))

    return px

# Parzen
def parzen(h, test):
  
    # Define an array for comparison
    px_0 = []
    px_1 = []
    px_2 = []
    result = []
    result_X = []


    for x in test:

        # Calculate the distance separately
        px_0 = get_px(x, train_0,h)
        px_1 = get_px(x, train_1, h)
        px_2 = get_px(x, train_2, h)

        # Find the largest index and sort
        result_X = np.argmax([px_0, px_1, px_2])
        result.append(result_X)

    return result


result_re = []

for h in np.arange(0.1, 0.5, 0.01):

    test = verify_X

    t = parzen(h, test)
    t = np.array(t)

    #Calculation accuracy
    result_re.append(np.sum(t == verify_Y.T) / len(t))

# Draw a graph to show the change of accuracy with K value
plt.title('Parzen Windows without cross-fold validation')
plt.plot(result_re)
plt.xlabel("h")
plt.ylabel("accuracy")
plt.show
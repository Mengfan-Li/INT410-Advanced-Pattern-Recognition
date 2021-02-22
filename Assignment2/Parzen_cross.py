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

# random the data
data = data.values
np.random.shuffle(data)

# test train verify data and use 5 cross validation

verify_1 =  data[0:30]
verify_2 =  data[30:60]
verify_3 =  data[60:90]
verify_4 =  data[90:120]
verify_5 =  data[120:150]

t0 = np.mat(t0)
t1 = np.mat(t1)
t2 = np.mat(t2)

train_0 = t0[0:30,:-1]
train_1 = t1[0:30,:-1]
train_2 = t2[0:30,:-1]
np.random.shuffle(train_0)
np.random.shuffle(train_1)
np.random.shuffle(train_2)

verify_1_X = verify_1[0:30,:-1]
verify_1_Y = verify_1[0:30,-1]
verify_2_X = verify_2[0:30,:-1]
verify_2_Y = verify_2[0:30,-1]
verify_3_X = verify_3[0:30,:-1]
verify_3_Y = verify_3[0:30,-1]
verify_4_X = verify_4[0:30,:-1]
verify_4_Y = verify_4[0:30,-1]
verify_5_X = verify_5[0:30,:-1]
verify_5_Y = verify_5[0:30,-1]

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

result_re_1 = []
result_re_2 = []
result_re_3 = []
result_re_4 = []
result_re_5 = []
aaa = []
result = []

for h in np.arange(0.1, 1, 0.01):

    """
    Calculate the accuracy of each cross-validation
    """
    test = verify_1_X
    t = parzen(h, test)
    t = np.array(t)
    result_re_1.append(np.sum(t == verify_1_Y.T) / len(t))
    
    test = verify_2_X
    t = parzen(h, test)
    t = np.array(t)
    result_re_2.append(np.sum(t == verify_2_Y.T) / len(t))
    
    test = verify_3_X
    t = parzen(h, test)
    t = np.array(t)
    result_re_3.append(np.sum(t == verify_3_Y.T) / len(t))
    
    test = verify_4_X
    t = parzen(h, test)
    t = np.array(t)
    result_re_4.append(np.sum(t == verify_4_Y.T) / len(t))
    
    test = verify_5_X
    t = parzen(h, test)
    t = np.array(t)
    result_re_5.append(np.sum(t == verify_5_Y.T) / len(t))

aaa = np.vstack((result_re_1,result_re_2,result_re_3,result_re_4,result_re_5))

for  i in range(40):
    bbb = 0
    bbb = aaa[0,i]+aaa[1,i]+aaa[2,i]+aaa[3,i]+aaa[4,i]
    result.append(bbb/5)

# Calculate the final accuracy rate and save it in the array
aaa = np.vstack((result_re_1,result_re_2,result_re_3,result_re_4,result_re_5))

for  i in range(40):
    bbb = 0
    bbb = aaa[0,i]+aaa[1,i]+aaa[2,i]+aaa[3,i]+aaa[4,i]
    result.append(bbb/5)

plt.title('Parzen Windows with cross-fold validation')
plt.plot(result)
plt.xlabel("h")
plt.ylabel("accuracy")
plt.show
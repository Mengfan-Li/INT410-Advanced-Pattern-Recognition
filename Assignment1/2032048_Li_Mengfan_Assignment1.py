import random
import pandas as pd
import numpy as np

# Label the pattern
iris_lable = {0: 'Iris-setosa',
              1: 'Iris-versicolor',
              2: 'Iris-virginica'}


# set a random seed, let the random stabilization
random.seed(17)


# random number generation
def rand(a, b):
    return (b - a) * random.random() + a


# Generate [i * j] matrix, default is zero matrix
def makeMatrix(i, j, fill=0.0):
    m = []
    for i in range(i):
        m.append([fill] * j)
    return m


#  tanh
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


#  tanh'(x)
def dtanh(x):
    return 1 - tanh(x) * tanh(x)


# the there layers NN
class T_L_Network:

    # input_data   ---> input layer points
    # hidden_data  ---> hidden layer points
    # output_data  ---> output layer point
    def __init__(self, input_data, hidden_data, output_data):

        self.input_data = input_data + 1  # to add a offset node
        self.hidden_data = hidden_data + 1  # to add a offset node
        self.output_data = output_data

        # activation the network all point, let them be float
        self.ainput = [1.0] * self.input_data
        self.ahidden = [1.0] * self.hidden_data
        self.aoutput = [1.0] * self.output_data

        # set up a weight matrix
        self.winput = makeMatrix(self.input_data, self.hidden_data)
        self.woutput = makeMatrix(self.hidden_data, self.output_data)
        # set the weight random ---> [(-(6/(i + 1 + j)) ** 0.5 ,(6/(i + 1 + j)) ** 0.5)]
        for i in range(self.input_data):
            for j in range(self.hidden_data):
                self.winput[i][j] = rand(-(6 / (i + 1 + j)) ** 0.5, (6 / (i + 1 + j)) ** 0.5)
        for j in range(self.hidden_data):
            for k in range(self.output_data):
                self.woutput[j][k] = rand(-(6 / (i + 1 + j)) ** 0.5, (6 / (i + 1 + j)) ** 0.5)

    def data_updata(self, inputs):

        # activation the input layer
        for i in range(self.input_data - 1):
            self.ainput[i] = inputs[i]

        # activation the hidden layer
        for j in range(self.hidden_data):
            sum = 0.0
            for i in range(self.input_data):
                sum = sum + self.ainput[i] * self.winput[i][j]
            self.ahidden[j] = tanh(sum)

        # activation the output layer
        for k in range(self.output_data):
            sum = 0.0
            for j in range(self.hidden_data):
                sum = sum + self.ahidden[j] * self.woutput[j][k]
            self.aoutput[k] = tanh(sum)

        return self.aoutput[:]

    # backpropagation
    def BP(self, targets, Learn_rate):

        # calculate the error of the output layer
        output_error_deltas = [0.0] * self.output_data

        for k in range(self.output_data):
            error = targets[k] - self.aoutput[k]
            output_error_deltas[k] = dtanh(self.aoutput[k]) * error

        # calculate the error of the hidden layer
        hidden_deltas = [0.0] * self.hidden_data

        for j in range(self.hidden_data):
            error = 0.0
            for k in range(self.output_data):
                error = error + output_error_deltas[k] * self.woutput[j][k]
                hidden_deltas[j] = dtanh(self.ahidden[j]) * error

        # update the output layer weight
        for j in range(self.hidden_data):
            for k in range(self.output_data):
                change = output_error_deltas[k] * self.ahidden[j]
                self.woutput[j][k] = self.woutput[j][k] + Learn_rate * change

        # update the input layer weight
        for i in range(self.input_data):
            for j in range(self.hidden_data):
                change = hidden_deltas[j] * self.ainput[i]
                self.winput[i][j] = self.winput[i][j] + Learn_rate * change

        # calculation error
        error += 0.5 * (targets[k] - self.aoutput[k]) ** 2
        # print(error)
        return error

    # train module
    def train(self, patterns, train_number, Learn_rate=0.001):
        for i in range(train_number):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.data_updata(inputs)
                error = error + self.BP(targets, Learn_rate)

    # test module
    def test(self, patterns):
        count = 0
        for p in patterns:
            target = iris_lable[(p[1].index(1))]
            result = self.data_updata(p[0])
            index = result.index(max(result))
            count += (target == iris_lable[index])
        accuracy = float(count / len(patterns))
        print('This model accuracy is:' + str(accuracy))


def iris_model():
    # read the iris.data
    data = []
    fresh = pd.read_csv('iris.data')
    fresh.shape
    fresh_data = fresh.values
    fresh_data_feature = fresh_data[0:, 0:4]

    # add the feature to the data
    for i in range(len(fresh_data_feature)):
        feature = []
        feature.append(list(fresh_data_feature[i]))
        if fresh_data[i][4] == 'Iris-setosa':
            feature.append([1, 0, 0])
        elif fresh_data[i][4] == 'Iris-versicolor':
            feature.append([0, 1, 0])
        else:
            feature.append([0, 0, 1])
        data.append(feature)

    # let the data random
    random.shuffle(data)

    # 0:99     ---> train
    # 100:150  ---> test
    training = data[0:119]
    test = data[120:]

    # input layer  have 4 points
    # hidden layer have 20 points
    # output layer have 3 points
    Model = T_L_Network(4, 20, 3)

    # train number
    Model.train(training, train_number = 200)

    Model.test(test)


# main function
if __name__ == '__main__':
    iris_model()

import pickle, gzip
import numpy as np
import time
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt

# load the dataset
def load_data():
    f = gzip.open("mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    f.close()
    return [train_set, valid_set, test_set]

# Define loss function
def loss_function(p_y_given_x, y):
    return -np.mean(np.log(p_y_given_x)[np.arange(y.shape[0]), y])

# Get the predicted label of each sample
def pred_num_lable(p_y_given_x):
    y_pred = []
    for i in range(p_y_given_x.shape[0]):
        max_index = np.argwhere(p_y_given_x[i] == np.max(p_y_given_x[i]))
        y_pred.append(max_index[0,0])
    return np.array(y_pred)

# Count the number of error samples
def error_num(y_pred, y):
    # print(y[:50])
    return np.sum((y_pred == y) == False)

# Construct the sample real label matrix
def truth_label_matrix(y):
    matrix = np.zeros((len(y), 10))
    for i in np.arange(len(y)):
        matrix[i, y[i]] = 1
    return matrix

# Sigmoid
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
# Sigmoid derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# tanh
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - (np.tanh(x) ** 2)



def hidden_layer(n_input, n_output, hidden_layer_num, rand):
    temp = []
    try:
        if len(hidden_layer_num) == 1:
            w_0 = np.asarray(rand.uniform(low=-np.sqrt(6. / (n_input + hidden_layer_num[0])),
                                          high=np.sqrt(6. / (n_input + hidden_layer_num[0])),
                                          size=(n_input, hidden_layer_num[0])))
            w_0 *= 4
            temp.append(w_0)
            w_1 = np.asarray(rand.uniform(low=-np.sqrt(6. / (hidden_layer_num[0] + n_output)),
                                          high=np.sqrt(6. / (hidden_layer_num[0] + n_output)),
                                          size=(hidden_layer_num[0], n_output)))
            w_1 *= 4
            temp.append(w_1)
        else:
            w_0 = 4 * np.asarray(rand.uniform(low=-np.sqrt(6. / (n_input + hidden_layer_num[0])),
                                              high=np.sqrt(6. / (n_input + hidden_layer_num[0])),
                                              size=(n_input, hidden_layer_num[0])))
            temp.append(w_0)
            for i in range(len(hidden_layer_num) - 1):
                temp.append(4 * np.asarray(rand.uniform(low=-np.sqrt(6. / (hidden_layer_num[i] + hidden_layer_num[i + 1])),
                                                     high=np.sqrt(6. / (hidden_layer_num[i] + hidden_layer_num[i + 1])),
                                                     size=(hidden_layer_num[i], hidden_layer_num[i + 1])))
                         )
            temp.append(4 * np.asarray(rand.uniform(low=-np.sqrt(6. / (hidden_layer_num[-1] + n_output)),
                                                 high=np.sqrt(6. / (hidden_layer_num[-1] + n_output)),
                                                 size=(hidden_layer_num[-1], n_output)))
                     )
    except:
        print("Warning: No input")
        pass
    return temp

# Calculate the derivative of |w|, needed when using L1 regularization
def sign(w):
    w = np.where(w > 0, 1, w)
    w = np.where(w >= 0, w, -1)
    return w


# Forward spread
def forward(x, W):
    layer_input = [x]                # input
    layer_output = [x]       # output

    # hidden layer
    for i in range(len(W) - 1):
        layer_input.append(np.dot(x, W[i]))
        temp_ = sigmoid(np.dot(x, W[i]))
        layer_output.append(temp_)
    # output layer
    layer_input.append(np.dot(temp_, W[-1]))
    layer_output.append((np.transpose(np.exp(np.dot(temp_, W[-1]))) / np.sum(np.exp(np.dot(temp_, W[-1])), axis=1)).T)

    return (layer_output, layer_input)


# Back propagation
def back(layer_output, layer_input, w, y, L1_reg, L2_reg):
    grad_w = [np.zeros(weight.shape) for weight in w]

    # Backpropagation
    delta = layer_output[-1] - y  # last layer, m*10
    grad_w[-1] = np.dot(np.transpose(layer_output[-2]), delta)
    for l in range(2, len(w) + 1):
        delta = np.dot(delta, w[-l + 1].transpose()) * sigmoid_derivative(layer_input[-l])  # other layer
        grad_w[-l] = np.dot(np.transpose(layer_output[-l - 1]), delta) / y.shape[0] + L1_reg * sign(w[-l]) + L2_reg * w[
            -l]

    return grad_w


# Parameter update
def gradient_update(y_truth_matrix, w_current, layer_output, layer_input, alpha, L1_reg, L2_reg):
    if len(w_current) > 1:
        # Calculate the gradient using backpropagation
        grad_w = back(layer_output, layer_input, w_current, y_truth_matrix, L1_reg, L2_reg)
        # Parameter update
        new_w = [w - nw * alpha for w, nw in zip(w_current, grad_w)]

    return new_w

# Gradient descent
def gradient_descent(x, y, w_initial, alpha, L1_reg, L2_reg):
    w = w_initial

    y_truth_matrix = truth_label_matrix(y)

    # Forward propagation to get model prediction value
    layer_output, layer_input = forward(x, w)
    probabilities = layer_output[-1]
    if L1_reg != 0.00:
        L1 = sum([abs(w_i).sum() for w_i in w])  # L1 norm
    else:
        L1 = 0
    if L2_reg != 0.00:
        L2_sqr = sum([(w_i ** 2).sum() for w_i in w])  # L2 norm
    else:
        L2_sqr = 0

    # Total loss = negative log likelihood cost + regularization penalty
    cost = loss_function(probabilities, y) + L1_reg * L1 + 0.5 * L2_reg * L2_sqr  # Error before weight w update

    # Update parameters
    w = gradient_update(y_truth_matrix, w, layer_output, layer_input, alpha, L1_reg, L2_reg)
    return (w, cost)

def model_validation(valid_set_x, valid_set_y, W):
    p_y_given_valid_x = forward(valid_set_x, W)[0][-1]
    valid_y_pred = pred_num_lable(p_y_given_valid_x)
    cost_valid = loss_function(p_y_given_valid_x, valid_set_y)
    error_num_valid = error_num(valid_y_pred, valid_set_y)
    return (cost_valid, error_num_valid)

def model_test(test_set_x, test_set_y, W):

    p_y_given_test_x = forward(test_set_x, W)[0][-1]
    test_y_pred = pred_num_lable(p_y_given_test_x)
    error_num_test = error_num(test_y_pred, test_set_y)
    test_precision = 1 - error_num_test / test_set_x.shape[0]
    return test_precision

def model_build(datasets, w_initial, alpha, epochs, threshold, batch_size, L1_reg, L2_reg):
    W = w_initial

    # Partition data set
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Divide the data set into smaller batches
    n_train_batches = train_set_x.shape[0] // batch_size

    # train model
    best_valid_cost = 0
    # Gradient descent iterative validation_frequency times,
    # verify the model performance once on the validation set
    validation_frequency = 100
    epoch = 0
    done_looping = False
    while (epoch < epochs) and (not done_looping):
        epoch += 1
        for batch_index in range(n_train_batches):
            x = train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size]  # X per training
            y = train_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]  # Corresponding y

            W, cost_train = gradient_descent(x, y, W, alpha, L1_reg, L2_reg)

            cost_valid, error_num_valid = model_validation(valid_set_x, valid_set_y, W)
            this_validation_loss = error_num_valid / valid_set_x.shape[0]  # error rate

            num_iter = (epoch - 1) * n_train_batches + batch_index
            if num_iter % validation_frequency == 0:
                # Track performance changes on the validation set
                print("After gradient descent iteration %d times, Accuracy is：%f%%"
                      % (num_iter,(1 - this_validation_loss) * 100))

            # Determine whether to early stopping
            if abs(cost_valid - best_valid_cost) < threshold:
                done_looping = True
                print("The error on the validation set no longer decreases, and the model training ends")
                break
            else:
                best_valid_cost = cost_valid

    if not done_looping:
        print("The maximum number of epochs is reached, and the model training ends")




    # model test
    test_precision = model_test(test_set_x, test_set_y, W)
    print("The accuracy of the model on the test set is：%f%%" % (test_precision * 100))

    return W


def mlp(datasets):
    # Specify hyperparameters
    alpha = 0.015            # learning rate
    epochs = 1000           # epoch num
    threshold = 0.00001     # Gradient descent early stop threshold
    batch_size = 300        # batch size
    L1_reg = 0.00           # L1 regularization parameters
    L2_reg = 0.001          # L2 regularization parameters

    # Initialize weight w
    input_layer_num = 784  # input layer number
    output_layer_num = 10  # output layer number
    hidden_layer_num = [500]    # hidden layer number
    rand = np.random.RandomState(int(time.time()))
    W = hidden_layer(input_layer_num, output_layer_num, hidden_layer_num, rand)  # Initialize connection weight

    # Model building and tuning
    # w_result = model_build(datasets, W, alpha, epochs, threshold, batch_size, L1_reg, L2_reg)
    # np.savetxt('mlp_w_result1.txt',w_result[0], fmt='%0.8f')
    # np.savetxt('mpl_w_result2.txt', w_result[1], fmt='%0.8f')

    # use model test
    w_result0 = np.loadtxt('mlp_w_result1.txt', dtype=np.float32)
    w_result1 = np.loadtxt('mlp_w_result2.txt', dtype=np.float32)
    w_form_txt = [w_result0, w_result1]
    test_set_x, test_set_y = datasets[2]
    test_precision = model_test(test_set_x, test_set_y, w_form_txt)
    print("The accuracy of the model on the test set is：%f%%" % (test_precision * 100))


if __name__ == "__main__":

    datasets = load_data()
    starttime = timeit.default_timer()
    mlp(datasets)
    endtime = timeit.default_timer()

    print("Code running%.2fmin" % ((endtime - starttime) / 60.))



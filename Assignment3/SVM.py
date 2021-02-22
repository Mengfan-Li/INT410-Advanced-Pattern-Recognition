import pickle
import gzip
from sklearn import svm
import time
import timeit


def load_data():
    print("start load data")
    """
    Return pattern recognition data containing tuples of training data, verification data, and test data
    The training data contains 50,000 pictures, and the test data and verification data only contain 10,000 pictures
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    print("load dataset successful")
    return (training_data, validation_data, test_data)



def svm_baseline():
    print("start SVM ")
    starttime = timeit.default_timer()
    training_data, validation_data, test_data = load_data()

    # Pass the parameters of the training model
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)

    # clf = svm.SVC(C=100.0, kernel='poly', gamma=0.03)

    # # model train
    # clf.fit(training_data[0], training_data[1])
    # s = pickle.dumps(clf)
    # f = open('svm.model', "wb+")
    # f.write(s)
    # f.close()

    # test
    f2 = open('svm.model', 'rb')
    s2 = f2.read()
    clf1 = pickle.loads(s2)


    predictions = [int(a) for a in clf1.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))

    print("SVM end")
    print("The accuracy rate is %s." % (num_correct / len(test_data[1])))
    endtime = timeit.default_timer()
    print("Code running%.2fmin" % ((endtime - starttime) / 60.))


if __name__ == "__main__":
    svm_baseline()
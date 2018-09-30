import autodiff as ad
import train
import initializers as init

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path, PurePath, WindowsPath
from operator import itemgetter
import tensorflow as tf
from tensorflow.python.framework import ops
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from tf_utils import load_dataset, convert_to_one_hot, predict

from image_dataset_load import *





def softmax_with_cross_entropy1(m, n, y):
    matmul_value = np.dot(m, n)
    sftmax = np.exp(matmul_value) / np.sum(np.exp(matmul_value), axis=0)
    return -np.sum(y * np.log(sftmax), axis=0)


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def one_hot(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    C = tf.constant(C, name='C')

    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot





def random_mini_batches(X, Y, batch_size=10, seed=0):
    """
    Shuffle the input X and produce mini batches in size of batch_size

    Arguments:
    X - input.  dimension ( input features, no of examples )
    Y - labels
    batch_size - size of the mini-batch
    seed - seed to randomize the X before splitting it into mini-batches

    Returns:
    batches of X and Y split into size of batch_size
    """


    np.random.seed(seed)

    m = X.shape[1]
    permute = list(np.random.permutation(m))

    X = X[:, permute]
    Y = Y[:, permute]

    a = list(range(0, m, batch_size))
    b = list(range(batch_size, m, batch_size))
    b.append(m)

    for (start, end) in zip(a, b):
        yield X[:, start:end], Y[:, start:end]


def loadImages():
    X, Y = loadImgsLabels((r"test\1", 0), (r"test\2", 1), (r"test\3", 2) )
    X, Y = randomize_std(X, Y)

    num_classes = 3

    #train_X, train_Y, dev_X, dev_Y, test_X, test_Y = split_train_dev_test(X, Y, 700, 800)
    train_X, train_Y, test_X, test_Y = split_train_test(X, Y, 700)

    train_Y = one_hot(train_Y, num_classes)
    #dev_Y = one_hot(dev_Y, num_classes)
    test_Y = one_hot(test_Y, num_classes)

    train_Y = train_Y.reshape((num_classes, train_Y.shape[2]))
    #dev_Y = dev_Y.reshape((num_classes, dev_Y.shape[2]))
    test_Y = test_Y.reshape((num_classes, test_Y.shape[2]))

    #aa = loadImgs(r"test\3", )

    print("train_X shape, train_Y shape ", train_X.shape, train_Y.shape)
    #print(dev_X.shape, dev_Y.shape)
    #print("dev_X shape, dev_Y shape ", dev_X.shape, dev_Y.shape)
    print("test_X shape, test_Y shape ", test_X.shape, test_Y.shape)

    return train_X, train_Y, test_X, test_Y


def accuracy1(executor, Z3, test_X, test_Y, x):
    #executor = ad.Executor([Z3])
    z3_cost = executor.compute_value([Z3], feed_dict = {x : test_X})[0]
    #z3_cost = executor.run(feed_dict = {x : test_X})[0]

    z3_am = np.argmax(z3_cost, axis=0)
    y_m = np.argmax(test_Y, axis=0)

    diff = z3_am - y_m
    c = diff[z3_am == y_m]
    #print("c shape ", c.shape)
    print("accuracy ", c.shape[0] * 100/test_X.shape[1])


def test_run():
    train_X, train_Y, test_X, test_Y = loadImages()

    learn_rate = 0.005

    # w_val_init1 = np.random.randn(100, train_X.shape[0]) * np.sqrt(2/train_X.shape[0])
    # w_val_init2 = np.random.randn(20, 100) * np.sqrt(2/100)
    # w_val_init3 = np.random.randn(3, 20) * np.sqrt(2/20)

    # t_W1 = tf.get_variable("W1", [100, train_X.shape[0]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # t_W2 = tf.get_variable("W2", [20, 100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # t_W3 = tf.get_variable("W3", [3, 20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #
    # init_op = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     w_val_init1 = sess.run(t_W1)
    #     w_val_init2 = sess.run(t_W2)
    #     w_val_init3 = sess.run(t_W3)




    # b1 = np.zeros((100, 1))
    # b1 = ad.Variable(b1, name = "b1")
    # b2 = np.zeros((20, 1))
    # b2 = ad.Variable(b2, name = "b2")
    # b3 = np.zeros((3, 1))
    # b3 = ad.Variable(b3, name = "b3")

    b1 = ad.Variable(name = "b1", shape = (100, 1), initializer = init.zeros_initializer())
    b2 = ad.Variable(name = "b2", shape = (20, 1), initializer = init.zeros_initializer())
    b3 = ad.Variable(name = "b3", shape = (3, 1), initializer = init.zeros_initializer())


    #W1, W2, W3 = ad.Variable(w_val_init1, name = "W1"), ad.Variable(w_val_init2, name = "W2"), ad.Variable(w_val_init3, name = "W3"),
    x, labels = ad.placeholder(name = "x"), ad.placeholder(name = "labels")



    W1 = ad.Variable(name = "W1", shape = (100, train_X.shape[0]), initializer = init.xavier_initializer())
    W2 = ad.Variable(name = "W2", shape = (20, 100), initializer = init.xavier_initializer())
    W3 = ad.Variable(name = "W3", shape = (3, 20), initializer = init.xavier_initializer())

    Z1 = ad.matmul(W1, x) + b1
    A1 = ad.relu(Z1)
    Z2 = ad.matmul(W2, A1) + b2
    A2 = ad.relu(Z2)
    Z3 = ad.matmul(W3, A2) + b3

    # Z1 = ad.matmul(W1, x) + b1
    # A1 = ad.relu(Z1)
    # Z2 = ad.matmul(W2, A1) + b2

    cost = ad.reduce_mean( ad.softmax_with_cross_entropy(Z3, labels) )
    #cost = ad.reduce_mean( ad.sigmoid_with_cross_entropy(Z3, labels) )
    optimizer = train.GradientDescentOptimizer(learn_rate).minimize(cost)
    optimizer.extend([cost])
    print("optimizer length ", len(optimizer))

    #print(optimizer)

    executor = ad.Executor(optimizer)
    t_w1, t_b1, t_w2, t_b2, t_w3, t_b3, minibatch_cost = executor.run(feed_dict = {x : train_X, labels : train_Y})

    #accuracy1(executor, Z3, test_X, test_Y, x)

    # seed = 0
    # costs = []   # to track the costs
    # epochs = 70
    # mini_batch_size = 32
    #
    # for epoch in range(epochs):
    #     epoch_cost = 0
    #     seed += 1
    #
    #     for (batch_X, batch_Y) in random_mini_batches(train_X, train_Y, mini_batch_size, seed):
    #         curr_batch_size = batch_X.shape[1]
    #
    #         t_w1, t_b1, t_w2, t_b2, t_w3, t_b3, minibatch_cost = executor.run(feed_dict = {x : batch_X, labels : batch_Y})
    #         #print(t_w1.shape, t_b1.shape, t_w2.shape, t_b2.shape)
    #         epoch_cost += minibatch_cost/curr_batch_size
    #
    #     if epoch % 5 == 0:
    #         print("Cost after epoch %i: %f" % (epoch, epoch_cost))
    #
    #
    #
    # # W1, W2, W3 = ad.Variable(t_w1, name = "W1"), ad.Variable(t_w2, name = "W2"), ad.Variable(t_w3, name = "W3"),
    # # b1 = ad.Variable(t_b1, name = "b1")
    # # b2 = ad.Variable(t_b2, name = "b2")
    # # b3 = ad.Variable(t_b3, name = "b3")
    # #
    # # Z1 = ad.matmul(W1, x) + b1
    # # A1 = ad.relu(Z1)
    # # Z2 = ad.matmul(W2, A1) + b2
    # # A2 = ad.relu(Z2)
    # # Z3 = ad.matmul(W3, A2) + b3
    #
    # accuracy1(executor, Z3, train_X, train_Y, x)
    #
    # accuracy1(executor, Z3, test_X, test_Y, x)


if __name__ == '__main__':


     #test_simple()

     #test_gradients()

     #test_run()

     resize_images(r"test\Interior - Driver Side_5", dimension=(106, 79))
     resize_images(r"test\Interior - Passenger Side_6", dimension=(106, 79))

     # X, Y = loadImgsLabels([(r"test\size_80x80\1", 0), (r"test\size_80x80\2", 1), (r"test\size_80x80\3", 2)], (80, 80))
     # X, Y = randomize_std(X, Y)
     #
     # train_X, train_Y, test_X, test_Y = split_train_test(X, Y, 700)

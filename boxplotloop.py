
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from data_processing import *
from hyperparams import *
from classification import *


def Mloop(tx_rem, y_new):

    rng = np.random.default_rng()
    indices = rng.permutation(tx_rem.shape[0])
    index_split = 10000
    index = indices[index_split :]

    tx_reduced = tx_rem[index,:]
    y_reduced = y_new[index]
    ratio = 0.8
    y_tr, x_tr, y_te, x_te = split_data(y_reduced,tx_reduced,ratio)

    #A
    initial_w = np.zeros([x_tr.shape[1],1])   
    max_iters = 50
    gamma = 0.7
    w_optA, _ = mean_squared_error_gd(y_tr, x_tr, initial_w, max_iters, gamma)

    #B
    w_optB, _ = mean_squared_error_sgd(y_tr, x_tr, initial_w, max_iters, gamma)

    #C
    w_optC, _ = least_squares(y_tr, x_tr)

    #D
    best_lambda, _= cross_validation_demo_ridge_reg(y_tr, x_tr, 7, 4, initial_w, np.logspace(-4, 0, 50), 3, gamma, max_iters)
    w_optD, _ = ridge_regression(y_tr,x_tr,best_lambda)

    #E
    initial_w = np.zeros([x_tr.shape[1],1])   
    max_iters = 50
    gamma = 0.4
    w_optE,_ = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
    y_resultE = sigmoid(x_te@w_optE)
    y_resultE[y_resultE>0.5] = 1
    y_resultE[y_resultE<0.5] = 0
    accuracyE = get_only_accuracy(y_resultE, y_te)


    #F
    initial_w = np.ones([x_tr.shape[1],1])
    max_iters = 50
    gamma = 0.7
    best_lambda,_ = cross_validation_demo(y_tr, x_tr, 7, 4, initial_w, np.logspace(-4, 0, 50), 3, gamma, max_iters)
    w_optF,_ = reg_logistic_regression(y_tr, x_tr, best_lambda, initial_w, max_iters, gamma)
    y_resultF = sigmoid(x_te@w_optF)
    y_resultF[y_resultF>0.5] = 1
    y_resultF[y_resultF<0.5] = 0
    accuracyF = get_only_accuracy(y_resultF, y_te)

    """ TAKES TOO LONG
    #G
    K = 20
    y_resultKNN = np.zeros((y_te.shape[0],1))
    for i in range (y_te.shape[0]):
        _, Kindexes = get_Kneighbors(x_tr, K, x_te[i,:])
        _, new_prediction = get_Kprediction(Kindexes, K, y_tr)
        y_resultKNN[i] = new_prediction
    accuracyG = get_only_accuracy(y_resultKNN, y_te)
    print('Done G')"""

    #getting every accuracy
    accuracyA = simple_class(x_te, y_te, w_optA)
    accuracyB = simple_class(x_te, y_te, w_optB)
    accuracyC = simple_class(x_te, y_te, w_optC)
    accuracyD = simple_class(x_te, y_te, w_optD)

    return accuracyA, accuracyB, accuracyC, accuracyD, accuracyE, accuracyF
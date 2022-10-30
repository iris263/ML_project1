## ******************************************************************************************************
#  IMPLEMENTATIONS OF REQUIRED FUNCTIONS
# this ver passed all tests

## ******************************************************************************************************

import numpy as np
import matplotlib.pyplot as plt


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    returns optimal weights, and mse.
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        w -= gamma * compute_mse_gradient(y,tx,w)
    
    return w, compute_mse_loss(y,tx,w)



def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
    returns optimal weights, and mse.

    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        w -= gamma * compute_stoch_mse_gradient(y,tx,w)

    return w, compute_mse_loss(y,tx,w)



def least_squares(y, tx):
    """Calculate the least squares solution.
    returns optimal weights, and mse.
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    
    A = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(A,b)
    return w, compute_mse_loss(y,tx,w)



def ridge_regression(y,tx,lambda_):
    """implement ridge regression.
    returns optimal weights, and mse.
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    A = tx.T@tx + 2*tx.shape[0]*lambda_*np.eye(tx.shape[1])
    b = tx.T@y
    w = np.linalg.solve(A,b)
    return w, compute_mse_loss(y,tx,w)



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y=0,1)

    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = w-gamma * calculate_logistic_gradient(y,tx,w)
    loss = calculate_logistic_loss(y,tx,w)
    return w, loss



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic Ridge regression using gradient descent (y=0,1)

    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    w = initial_w
    for n_iter in range(max_iters):
        penalized_gradient = calculate_logistic_gradient(y,tx,w)+ 2 * lambda_ * w
        w -= gamma * penalized_gradient
    loss = calculate_logistic_loss(y,tx,w) 
    # convention: loss is always without the penalty term
    return w, loss


## ******************************************************************************************************
#  auxiliary functions 

## ******************************************************************************************************

def compute_mse_loss(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return (1/(2*len(y))) * np.sum(np.square(y - tx@w))


def compute_mse_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    return - (tx.T@(y - tx@w))/len(y)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_stoch_mse_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    batch_size=1
    for value in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True): 
        grad = compute_mse_gradient(value[0],value[1],w)
    
    return grad


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1/(1+np.exp(-t))


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss
    """
    # assert y.shape[0] == tx.shape[0]
    # assert tx.shape[1] == w.shape[0]

    loss = -(1/len(y)) * ( y.T @ np.log(sigmoid(tx@w)) + (1-y).T @ np.log(1-sigmoid(tx@w)) )
    loss = np.sum(loss)  # to avoid nested 1-D arrays
    return loss


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss.
    
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a vector of shape (D, 1)
    """
    return (1/len(y))*tx.T@(sigmoid(tx@w)-y)


def calculate_stoch_logistic_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    batch_size=1
    for value in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True): 
        grad = calculate_logistic_gradient(value[0],value[1],w)
    
    return grad


def stoch_reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic Ridge regression using stochastic gradient descent (y=0,1)

    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    w = initial_w
    for n_iter in range(max_iters):
        penalized_gradient = calculate_stoch_logistic_gradient(y,tx,w)+ 2 * lambda_ * w
        w -= gamma * penalized_gradient
    loss = calculate_logistic_loss(y,tx,w) 
    # convention: loss is always without the penalty term
    return w, loss




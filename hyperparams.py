import numpy as np
import matplotlib.pyplot as plt
from implementations import *


#Hyperparameters optimization


def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k,initial_w, lambda_, degree ,gamma, max_iters):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)"""

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)] #comprendre cette ligne ?
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    

    w, loss_tr = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
   
    loss_te =  calculate_logistic_loss(y_te, x_te, w)
    
    return loss_tr, loss_te


def cross_validation_demo(y, x, k_fold,k, initial_w, lambdas, degree ,gamma, max_iters):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    seed = 12
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    max_iters = max_iters
    
    for lambda_ in lambdas : 
        rmse_tr_k = []
        rmse_te_k = []

        for  k in range(k_fold): 
                
            tr,te = cross_validation(y, x, k_indices,k, initial_w,lambda_, degree ,gamma, max_iters)
            rmse_tr_k.append(tr)
            rmse_te_k.append(te)
        
        rmse_tr.append(np.mean(rmse_tr_k))
        rmse_te.append(np.mean(rmse_te_k))
    
    best_rmse = np.min(rmse_tr)
    best_ind= np.argmin(rmse_tr)
    best_lambda = lambdas[best_ind]
    

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    return best_lambda, best_rmse


#Use poly and find best degree

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
        
    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    N = x.shape[0]
    poly = np.square(x)
    poly = np.reshape(poly, [N,1])
    if degree == 2: 
        return poly
    else : 
        for deg in range(3, degree+1):
            poly = np.c_[poly, np.power(x, deg)]
        return poly

def best_degree_selection(y,x,degrees, k_fold, initial_w, lambdas, gamma,max_iters,seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    #vary degree
   
    for degree in degrees:
        # cross validation
        phi = build_poly(x,degree)
        rmse_te = []
        for lambda_ in lambdas:
            initial_w = ridge_regression(y,phi,lambda_)[0]
            rmse_te_tmp = []
            for k in range(k_fold):
                
                _, loss_te= cross_validation(y, phi, k_indices,k, initial_w,lambda_, degree ,gamma, max_iters)
                rmse_te_tmp.append(loss_te)
          
            rmse_te.append(np.mean(rmse_te_tmp))
        
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree]

def phi_optimized(y,x,degrees,P, k_fold, initial_w, lambdas, gamma,max_iters,columns_to_expand,seed = 1) : 
    #Calcul du meilleur degré pour chaque colonne
    degrees_table = []
    gamma = 0.4
    nb_data_used = x.shape[0]
    phi = x
    i = 0
    for column in columns_to_expand : #ne prend pas en compte la première colonne de x, qui est la colonne de 1 VERIFIER QU'ON VA BIEN JUSQU4A LA DERNIERE
        
        degrees_table.append(best_degree_selection(y,x[:,column],degrees, k_fold, initial_w, lambdas, gamma,max_iters,seed = 1))
        if (degrees_table[i] >1) : 
            poly_x = build_poly(x[:,column], degrees_table[i])
            phi = np.c_[ phi, poly_x ]
        i=i+1

    return phi, degrees_table




# cross validation for ridge regression

def cross_validation_ridge_reg(y, x, k_indices,k, initial_w,lambda_, degree ,gamma, max_iters):
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)] 
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]

    w, loss_tr = ridge_regression(y_tr, x_tr, lambda_)
   
    loss_te =  calculate_logistic_loss(y_te, x_te, w)
    loss_te =  compute_mse_loss(y_te, x_te, w)

    return loss_tr, loss_te



def cross_validation_demo_ridge_reg(y, x, k_fold, k, initial_w, lambdas, degree ,gamma, max_iters):

    seed = 12
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    max_iters = max_iters
    
    for lambda_ in lambdas : 
        rmse_tr_k = []
        rmse_te_k = []

        for  k in range(k_fold): 
                
                tr,te = cross_validation_ridge_reg(y, x, k_indices,k, initial_w,lambda_, degree ,gamma, max_iters)
                rmse_tr_k.append(tr)
                rmse_te_k.append(te)
        
        rmse_tr.append(np.mean(rmse_tr_k))
        rmse_te.append(np.mean(rmse_te_k))
    
    best_rmse = np.min(rmse_tr)
    best_ind= np.argmin(rmse_tr)
    best_lambda = lambdas[best_ind]
    

    return best_lambda, best_rmse

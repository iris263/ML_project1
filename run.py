import numpy as np
from helpers import *
from implementations import *
from data_processing import *
from hyperparams import *
from classification import *

# loading train data 
yb, input_data, ids = load_csv_data("train.csv")
dimensions = np.shape(input_data)
N = dimensions[0]
P = dimensions[1]
yb = np.reshape(yb,[N,1])
yb[yb==-1] = 0


tx = data_replace(input_data)

#Standardize each feature according to its type of distribution

indices_min_max =[3,11,12,22,26]
indices_gaussian =[0,1,6,8,13,14,16,17,24,27]
indices_angles = [15,18,20,25,28]
indices_gaussian_log = [2,5,7,9,10,19]

normalize (tx, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max)

tx = np.delete(tx,29,1)
tx = np.delete(tx,23,1)
tx = np.delete(tx,21,1)
tx = np.delete(tx,4,1)

tx = add_w0(tx,tx.shape[0])   


x_tr = tx
y_tr = yb

initial_w = np.zeros([x_tr.shape[1],1])   
max_iters = 100
gamma = 0.7
w_opt,loss = logistic_regression(y_tr,x_tr,initial_w,max_iters,gamma)

print(loss)
print(w_opt.shape)

#For reg logistic regression

#initial_w = np.ones([x_tr.shape[1],1])
#max_iters = 20
#best_lambda, best_rmse = cross_validation_demo(y_tr, x_tr, 7, 4,initial_w,  np.logspace(-4, 0, 70), 3 ,gamma, max_iters )
#w_opt2,loss2 = reg_logistic_regression(y_tr, x_tr, best_lambda, initial_w, max_iters, gamma)
#print(loss2)

y_test, x_test, ids = load_csv_data("test.csv")
dimensions_te = np.shape(x_test)
N = dimensions_te[0]
P = dimensions_te[1]
y_test = np.reshape(y_test,[N,1])
y_test[y_test==-1] = 0

# Replace by mean/median

x_te = data_replace(x_test)
y_te = y_test

#Standardize each feature according to its type of distribution

indices_min_max =[3,11,12,22,26]
indices_gaussian =[0,1,6,8,13,14,16,17,24,27]
indices_angles = [15,18,20,25,28]
indices_gaussian_log = [2,5,7,9,10,19]

normalize (x_te, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max)

x_te = np.delete(x_te,29,1)
x_te = np.delete(x_te,23,1)
x_te = np.delete(x_te,21,1)
x_te = np.delete(x_te,4,1)

x_te= add_w0(x_te,x_te.shape[0]) 

y_pred = sigmoid(x_te@w_opt)
y_pred[y_pred>0.5] = 1
y_pred[y_pred<0.5] = -1

create_csv_submission(ids, y_pred, "predictions")
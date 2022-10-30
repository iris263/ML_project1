import numpy as np


def data_replace(tx):
    """replace missing data in tx in a feature (column) by the mean or median of the feature if more than half the data is missing
    Args:
        tx: shape=(N,P) N is the number of samples, D is the number of features
    
    Returns:
        tx: modified tx 
    """

    idx_incomplete_points = np.nonzero(tx[:,4]==-999)
    tx_rem = np.delete(tx,idx_incomplete_points,0)
    
    means = np.mean(tx_rem, axis=0)
    meds = np.median(tx_rem, axis=0)

    #calculate mean for the first column(otherwise, takes into account -999 in calculation)
    idx_incomplete_points = np.nonzero(tx[:,0]==-999)
    tx_rem= np.delete(tx,idx_incomplete_points,0)
    mean_first = np.mean(tx_rem[:,0])
    first = tx[:,0]
    first[first==-999] = mean_first
    tx[:,0] = first

    for i in range(1,tx.shape[1]):
        feature = tx[:,i]
        
        nan = feature[feature == -999] 
        if (len(nan)>= tx.shape[1]/2) : 
            feature[feature==-999] = meds[i] 
        else: 
            feature[feature==-999] = means[i] 
        tx[:,i] = feature
  
    return tx


def split_data(y,tx,ratio):
    """split a data set in a training part and a test part
    with a given ratio
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        ratio: scalar, indicates amount of training data
    
    Returns:
        y_tr, x_tr: training data
        y_te, x_te: test data
    """
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]

    x_tr = tx[index_tr]
    x_te = tx[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return y_tr, x_tr, y_te, x_te

def add_w0(tx,N):
    tx = np.concatenate((np.ones([N,1]),tx),axis=1)  
    return tx

def data_removed(y,tx):
    """remove all points (rows) with missing data
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) N is the number of samples, D is the number of features
    
    Returns:
        y_new: reduced y
        tx_new: reduced x
    """
    idx_incomplete_points = np.nonzero(tx[:,4]==-999)
    tx_new = np.delete(tx,idx_incomplete_points,0)
    y_new = np.delete(y,idx_incomplete_points)
    idx_incomplete_points = np.nonzero(tx_new[:,0]==-999)
    tx_new = np.delete(tx_new,idx_incomplete_points,0)
    y_new = np.delete(y,idx_incomplete_points)
    y_new = np.reshape(y_new,[len(y_new),1])

    return y_new, tx_new

def normalize_log_gaussian(tx_rem, indices_gaussian_log) : 
    for x in indices_gaussian_log :
        c = tx_rem[:,x]
        c = np.log(tx_rem[:,x])
        mean = np.mean(c, axis=0)
        std_dev = np.std(c, axis=0)
        tx_rem[:,x] = (c -mean*np.ones(np.shape(c))) / std_dev
    #voir si modifie direct ou si doit faire retour ? devrait être modifié comme fait référence automatiquement non ?
        
def normalize_angles(tx_rem, indices_angles):
    for x in indices_angles : 
        c = tx_rem[:,x]
        """mean = np.mean(c, axis=0)
    
        tx_rem[:,x] = (c - mean) 
        tx_rem[:,x] = tx_rem[:,x]/np.max(np.abs(tx_rem[:,x]))"""
        tx_rem[:,x]  = np.cos(tx_rem[:,x])
        
def normalize_gaussian(tx_rem, indices_gaussian) :
    for x in indices_gaussian :
        c = tx_rem[:,x]
        
        mean = np.mean(c, axis=0)
        std_dev = np.std(c, axis=0)
        tx_rem[:,x] = (c -mean*np.ones(np.shape(c))) / std_dev

def normalize_min_max(tx_rem,indices_minmax):
    for x in indices_minmax :
        c = tx_rem[:,x]
        min = np.min(c, axis=0)
        max = np.max(c, axis=0)
        tx_rem[:,x] = (c-min) / (max-min)

def normalize (tx_rem, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max) : 
    normalize_log_gaussian(tx_rem, indices_gaussian_log) 
    normalize_angles(tx_rem, indices_angles)
    normalize_gaussian(tx_rem, indices_gaussian)
    normalize_min_max(tx_rem,indices_min_max)
   

     
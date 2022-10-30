# # ***************************************************
#  IMPLEMENTATIONS OF  ADDITIONNAL CLASSIFICATION FUNCTION

# # ***************************************************

import numpy as np
import matplotlib.pyplot as plt


def simple_class(x_te, y_te, w_opt):
    y_result = x_te@w_opt
    y_result[y_result>0.5] = 1
    y_result[y_result<0.5] = 0
    accuracy = get_only_accuracy(y_result, y_te)
    return accuracy



def get_only_accuracy(y_result, y_te):
    difference = (y_result-y_te)
    good_guess = difference[difference==0]
    bad_guess = difference[difference!=0]
    accuracy = len(good_guess)/(len(good_guess)+len(bad_guess))
    return accuracy


def get_accuracy(y_result, y_te, score):
    """Checks whether prediction are accurate by compraing with y_te
    
    Args: 
        y_results: shape=(K,)
        y_te: shape=(N,). Known predictions of test samples
    
    Returns:
        accuracy: scalar = (TP+TN)/(TP+TN+FP+FN)
        precision: scalar = TP/(TP+FP)
        recall: scalar = TP/(TP+FN)
        auc: scalar = area under ROC curve 
    """ 
    
    difference = (y_result-y_te)
    good_guess = difference[difference==0]
    bad_guess = difference[difference!=0]
    accuracy = len(good_guess)/(len(good_guess)+len(bad_guess))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(y_result.shape[0]):
        if difference[i] == 1:
            FP +=1
        if difference[i] == -1:
            FN +=1
        else :
            if y_result[i] == 1:
                TP +=1
            else:
                TN +=1       
                
    print("Good and bad guesses: %.f  %.f" %(len(good_guess), len(bad_guess)))
    
    precision = TP/(TP+FP)  
    recall = TP/(TP+FN)
    FPR,TPR,auc = get_auc(score, y_te)
    
    print("How well our model can classify binary outcomes: ")
    print("Accuracy: %.3f" %(accuracy))
    print("Precision: %.3f" %(precision))
    print("Recall: %.3f" %(recall))
    print("AUC: %.3f" %(auc))
    roc_visualization(FPR,TPR,auc)

    return accuracy, precision, recall, auc




def get_auc(score, y_result):
    """Computes FPR, TPR and AUC
    
    Args: 
        score: shape=(N,). Contains values between 0 and 1 for each sample 
        y_result: shape=(N,). Contains 0 or 1 for each sample
    
    Returns:
        FPR: shape=(len(thresholds),). False Positive Rate
        TPR: shape=(len(thresholds),). True Positive Rate
        AUC: Area under ROC curve, scalar
    """ 

    FPR = [] # false positive rate
    TPR = [] # true positive rate
    # Iterate thresholds from 0.0 to 1.0
    thresholds = np.arange(0.0, 1.01, 0.001)

    # get number of positive and negative results
    P = sum(y_result)
    N = len(y_result) - P

    # iterate on thresholds and determine fraction of true positives and false positives found
    for thresh in thresholds:
        FP=0
        TP=0
        thresh = round(thresh,2) 
        for i in range(len(score)):
            if (score[i] >= thresh):
                if y_result[i] == 1:
                    TP += 1
                if y_result[i] == 0:
                    FP += 1            
        FPR = np.append(FPR,FP/N)
        TPR = np.append(TPR, TP/P)

    #computing Arean Under Curve using the trapezoidal method
    auc = -1 * np.trapz(TPR, x=FPR)
    
    return FPR,TPR,auc



def roc_visualization(FPR,TPR,auc):

    plt.plot(FPR, TPR, marker='.', color='darkorange', label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label = 'No Discrimination')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve, AUC = %.2f'%auc)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('AUC_example.png')
    plt.show()







# functions used for KNN

def get_Kneighbors(x_tr, K, new_sample):
    """Finds K nearest neighbors using euclidian distance
    
    Args:
        x_tr: shape=(N,P). N is the number of samples and D is the number of features
        K: scalar, number of nearest neighbors 
        new_sample: shape(N,P). New sample to be classified

    Returns:
        Kneighbors: shape=(K,)
        Kindexes: shape=(K,) 
    """
    samples = np.shape(x_tr)[0] 
    neighbors = np.zeros((samples,1))
    Kneighbors = np.zeros((K,1))
    Kindexes = np.zeros((K,1))
    
    for i in range (samples):
        #computes euclidean distance between two samples
        neighbors[i] = np.linalg.norm(x_tr[i,:] - new_sample)  

    for j in range (K):
        Kneighbors[j] = min(neighbors)
        idx = np.argmin(neighbors)
        #remove the last smallest value for the next iteration 
        Kindexes[j] = idx 
        neighbors = np.delete(neighbors, idx)
        
    return Kneighbors, Kindexes  


def get_Kprediction(Kindexes, K, y_tr):
    """Predicts y = 0 or 1 using the prediction of K-nearest neighbors
    
    Args:
        Kindexes: shape=(K,). Indexes of K nearest neighbors 
        K: scalar, number of nearest neighbors
        y_tr : shape(N,). Predictions of train samples
        
    Returns:
        predictions: shape(K,) 
        new_prediction: scalar = 0 or 1   
    """
    
    predictions = np.zeros((K,1))
    for m in range (K):
        predictions[m] = y_tr[np.int(Kindexes[m])] 
        
    scoreKNN = np.mean(predictions)
    if scoreKNN >= 0.5:
        new_prediction = 1
    else:
        new_prediction = 0
        
    return scoreKNN, new_prediction



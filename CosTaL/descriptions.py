import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from itertools import compress
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import fowlkes_mallows_score, homogeneity_completeness_v_measure

def return_Hf1_score(ytrue, ypred):
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)
    
    flag = ~np.isnan(ytrue)
    ytrue = list(compress(ytrue,flag))
    ypred = list(compress(ypred,flag))

    conf =  pd.crosstab(np.array(ytrue, dtype=object),  np.array(ypred, dtype=object))

    f1_matrix = np.empty(conf.shape)
    truth_all = list(conf.sum(axis =1))
    detected_all = list(conf.sum(axis =0))

    for i in range(conf.shape[0]): # true
        for j in range(conf.shape[1]): # predic
            true_positive = conf.iloc[i,j]
            truth = truth_all[i]
            detected = detected_all[j]
            
            if detected: precision_ij = true_positive/detected
            else: precision_ij = 0
            if truth: recall_ij = true_positive/truth
            else: recall_ij = 0
            
            if (precision_ij + recall_ij ==0): f1_ij = 0
            else: f1_ij = 2*(precision_ij*recall_ij)/(precision_ij+recall_ij)
            f1_matrix[i][j] = f1_ij
    
    f1_matrix_cost = np.ones(f1_matrix.shape) - f1_matrix
    row_ind, col_ind = linear_sum_assignment(f1_matrix_cost)
    true_to_clust = dict(zip(row_ind, col_ind))
    F1 = [f1_matrix[i, true_to_clust.get(i)] if i in row_ind else 0 for i in range(f1_matrix.shape[0])]
    return np.mean(F1)

def return_Ff1_score(ytrue, ypred):
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)

    flag = ~np.isnan(ytrue)
    ytrue = list(compress(ytrue,flag))
    ypred = list(compress(ypred,flag))
    conf = pd.crosstab(np.array(ytrue, dtype=object),  np.array(ypred, dtype=object))
    
    f1_matrix = np.empty(conf.shape)
    truth_all = list(conf.sum(axis =1))
    detected_all = list(conf.sum(axis =0))
    
    for i in range(conf.shape[0]): # true
        for j in range(conf.shape[1]): # predic
            true_positive = conf.iloc[i,j]
            truth = truth_all[i]
            detected = detected_all[j]
            
            if detected: precision_ij = true_positive/detected
            else: precision_ij = 0
            if truth: recall_ij = true_positive/truth
            else: recall_ij = 0
            
            if (precision_ij + recall_ij ==0): f1_ij = 0
            else: f1_ij = 2*(precision_ij*recall_ij)/(precision_ij+recall_ij)
            f1_matrix[i][j] = f1_ij
       
    ytrue_sizes = np.unique(ytrue, return_counts=True)[1]
    ytrue_percents = ytrue_sizes/sum(ytrue_sizes)
    return sum(np.multiply(ytrue_percents, f1_matrix.max(axis = 1)))

def return_scores(ytrue, ypred):
    """
    Return 3 kinds of scores for the evaluation of clustering algorithms

    Args:
        ytrue: gold standard clustering labels
        ypred: labels from clustering algorithm

    Returns:
        AMI score, ARI score, F1 score(FlowCAPI method), F1 score(Hunggarian method)

    """
    warnings.filterwarnings('ignore')  # avoid verbose numpy warnings
    if isinstance(ytrue, (np.ndarray, pd.Series)):
        ytrue = list(ytrue)
    elif isinstance(ytrue, list):
        if any(isinstance(x, list) for x in ytrue):
            ytrue = ['_'.join(list(map(str, item))) for item in ytrue]
        pass
    else: raise('Not support input format for input true labels')
    
    if isinstance(ypred, (np.ndarray, pd.Series)):
        ypred = list(ypred)
    elif isinstance(ypred, list):
        if any(isinstance(x, list) for x in ypred):
            ypred = ['_'.join(list(map(str, item))) for item in ypred]
        pass
    else: raise('Not support input format for input true labels')
    
    ytrue = pd.Series(ytrue)
    ypred = pd.Series(ypred)

    subset_index = ytrue.notna()
    labels = ytrue[subset_index]

    communities = ypred[subset_index]

    le = LabelEncoder()
    #df = pd.DataFrame({'original': le.fit_transform(labels), 'result': le.fit_transform(communities)}).astype('category')
    #original = df.original
    original = le.fit_transform(labels)
    #result = df.result
    result = le.fit_transform(communities)
    Homogeneity, Completeness, V_measure = homogeneity_completeness_v_measure(original, result)
    return (
        # AMI
        adjusted_mutual_info_score(original, result), 
        # ARI
        adjusted_rand_score(original, result), 
        # FF1
        return_Ff1_score(original, result), 
        # HF1
        return_Hf1_score(original, result),
        # Fowlkes_Mallows_Score
        fowlkes_mallows_score(original, result),
        # Homogeneity, Completeness, V_measure
        Homogeneity, Completeness, V_measure,
        # Singlets
        sum(np.unique(ypred, return_counts=True)[1]==1),
        # Small communities
        (sum(np.unique(ypred, return_counts=True)[1] <= 10) - sum(np.unique(ypred, return_counts=True)[1]==1)),
        # Decent communities
        (len(np.unique(ypred)) - sum(np.unique(ypred, return_counts=True)[1] <= 10)),
        # All communities
        len(np.unique(ypred))
        )

def get_f1_matrix(ytrue, ypred):
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)
    
    flag = ~np.isnan(ytrue)
    ytrue = list(compress(ytrue,flag))
    ypred = list(compress(ypred,flag))

    conf =  pd.crosstab(np.array(ytrue, dtype=object),  np.array(ypred, dtype=object))

    f1_matrix = np.empty(conf.shape)
    truth_all = list(conf.sum(axis =1))
    detected_all = list(conf.sum(axis =0))

    for i in range(conf.shape[0]): # true
        for j in range(conf.shape[1]): # predic
            true_positive = conf.iloc[i,j]
            truth = truth_all[i]
            detected = detected_all[j]
            
            if detected: precision_ij = true_positive/detected
            else: precision_ij = 0
            if truth: recall_ij = true_positive/truth
            else: recall_ij = 0
            
            if (precision_ij + recall_ij ==0): f1_ij = 0
            else: f1_ij = 2*(precision_ij*recall_ij)/(precision_ij+recall_ij)
            f1_matrix[i][j] = f1_ij
    return(f1_matrix)

def get_rare_f1(ytrue, ypred):
    ytrue = [str(i) for i in list(ytrue)]
    ypred = [str(i) for i in list(ypred)]
    le = LabelEncoder()
    original = le.fit_transform(ytrue)
    result = le.fit_transform(ypred)
    f1_matrix = get_f1_matrix(original, result)
    return(f1_matrix[1,:].max())

def calculate_sparsity(data):
    data = np.array(data)
    return(1.0 - np.count_nonzero(data)/float(data.size))
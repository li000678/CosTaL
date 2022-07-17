#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Function: 
# Author: Yijia Li

import os, sys, glob, time, uuid
import warnings
import numpy as np
import pandas as pd
import itertools
import subprocess
import igraph as ig
import leidenalg as la
from contextlib import closing
from multiprocessing import Pool
from itertools import repeat
from numba import njit, float64
from scipy import sparse as sp
from scipy.linalg import norm
from sklearn.preprocessing import normalize
from phenograph import bruteforce_nn # temproarly using phenograph on platforms other than Linux
from umap import umap_ as u

if __name__ == '__main__':
    def main():
        pass


def normalize_arcsinh(X, base = 5):
    """
    Reutrn the arcsinh transformed data. base = 5 for mass cytometry; base = 150 for flow cytometry.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse.csr.csr_matrix
        input matrix
    base : int, optional
        base of the transformation, which X will be divided by before arcsinh transformation, by default 5
    
    Returns
    -------
    same as X
        transformed data
    """
    return(np.arcsinh(X/base).copy())


def minus_1_normalize_arcsinh(X, noise_threshold = 1, base = 5):
    """To reduce the noise, set all the values no greater than 1 to 0, then reutrn the arcsinh transformed data. base = 5 for mass cytometry; base = 150 for flow cytometry.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse.csr.csr_matrix
        input data for transformation
    noise_threshold : int, optional
        noise level, by default 1
    base : int, optional
        base of the transformation, which X will be divided by before arcsinh transformation, by default 5
    """
    temp = np.array(np.clip(X - noise_threshold, a_min = 0, a_max = None))
    temp[np.where(np.all(np.isclose(temp, 0), axis=1))[0]] = 1.0
    return(normalize_arcsinh(temp, base=base))

def pp(X, platform = None):
    temp = type(X)
    if temp not in [pd.core.frame.DataFrame, np.ndarray]:
        if not sp.issparse(X):
            raise RuntimeError('Input datatype not supported!')
    if platform == 'mass':
        if temp is pd.core.frame.DataFrame:
            X = normalize_arcsinh(X.select_dtypes([np.number]).to_numpy(), base = 5)
        else:
            X = normalize_arcsinh(X, base = 5)
    elif platform == 'flow':
        if temp is pd.core.frame.DataFrame:
            X = normalize_arcsinh(X.select_dtypes([np.number]).to_numpy(), base = 150)
        else:
            X = normalize_arcsinh(X, base = 150)
    elif platform == 'spectral flow':
        if temp is pd.core.frame.DataFrame:
            X = normalize_arcsinh(X.select_dtypes([np.number]).to_numpy(), base = 6000)
        else:
            X = normalize_arcsinh(X, base = 6000)
            
    elif platform == 'scrnaseq':
        import scanpy as sc
        import anndata
        if temp is anndata._core.anndata.AnnData:
            X = X.X
        if sp.issparse(X):
            X = sp.csr_matrix(X)
        elif temp is pd.core.frame.DataFrame:
            X = X.select_dtypes([np.number]).to_numpy()
        elif temp is np.ndarray:
            pass
        else: 
            raise RuntimeError('Input data type not supported')
        data = anndata.AnnData(X= X)
        sc.pp.filter_cells(data, min_genes=200)
        sc.pp.filter_genes(data, min_cells=3)
        sc.pp.highly_variable_genes(data, n_top_genes= 2000, flavor= 'seurat_v3')
        sc.pp.log1p(data)
        X = data.X[:,data.var.highly_variable]
    else:
        if temp is pd.core.frame.DataFrame:
            X = X.select_dtypes([np.number]).to_numpy()
        else: pass
    return(X)

def save_as_ijv(filename, X):
    """
    Convert matrix X to coo sparse matrix and save as .ijv file for l2knng to readin the data and find neighbors

    Parameters
    ----------
    filename : string
        filename for the temp file
    X : numpy.ndarray or scipy.sparse.csr.csr_matrix
        matrix to be stored in the file
    """

    coo = sp.coo_matrix(X)
    i, j = coo.nonzero()
    v = coo.data
    with open(filename, 'w', encoding='ascii') as f:
        f.writelines('{} {} {}\n'.format(a, b, c) for a, b, c in zip(i, j, v))
    return


@njit(float64(float64, float64, float64), cache=True)
def tani_from_cos(a, b, cos_val):
    """
    Calculate tanimoto coefficient using cosine similarity and the norm of the cells

    Parameters
    ----------
    a : float
        norm of one cell
    b : float
        norm of one cell
    cos_val : float
        Cosine similarity between two cells

    Returns
    -------
    float
        calculated Tanimoto coefficient
    """
    d = a*b*cos_val
    if d ==0:
        return 0
    else:
        return (d/(a*a+b*b-d))


def build_graph(rows, columns, weights, shape):
    """
    Build a igraph.Graph from the pruned graph

    Parameters
    ----------
    rows : list
        list of row coordinates of the cells
    columns : list
        list of column coordinates of the cells
    weights : list
        list of Tanimoto coefficients as weights
    shape : tuple
        shape of the adjacency matrix, number of cells by number of cells

    Returns
    -------
    igraph.Graph
        converted graph to be the input of the Leiden algorithm
    """
    warnings.filterwarnings('ignore')
    csr = sp.csr_matrix((np.array(weights), (np.array(rows), np.array(columns))), shape=shape)
    #csr = (csr + csr.transpose()).multiply(0.5)
    csr = csr+csr.transpose() - csr.multiply(csr.transpose())
    csr.setdiag(0)
    csr = sp.triu(csr)
    csr.eliminate_zeros()
    warnings.resetwarnings()
    sources, targets = csr.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    G = ig.Graph(edges = edgelist, edge_attrs = {'weight': csr.data.tolist()}) 
    warnings.filterwarnings('default')
    return G

def build_graph_umap(rows, columns, weights, shape, nbr_num, set_op_mix_ratio=1.0, local_connectivity = 1.0):
    """
    Build a igraph.Graph from the pruned graph

    Parameters
    ----------
    rows : list
        list of row coordinates of the cells
    columns : list
        list of column coordinates of the cells
    weights : list
        list of Tanimoto coefficients as weights
    shape : tuple
        shape of the adjacency matrix, number of cells by number of cells

    Returns
    -------
    igraph.Graph
        converted graph to be the input of the Leiden algorithm
    """
    
    warnings.filterwarnings('ignore')
    distances = np.array(weights).reshape(shape[0], nbr_num).astype(np.float32)
    indices = np.array(columns).reshape(shape[0], nbr_num)
    sigmas, rhos = u.smooth_knn_dist(
        distances,
        float(nbr_num),
        local_connectivity=float(local_connectivity),
    )
    #sigmas, rhos = ct_smooth_knn_dist(distances, nbr_num)
    r, c, v, d = u.compute_membership_strengths(indices, distances, sigmas, rhos, return_dists = False)
    
    result = sp.coo_matrix((v, (r, c)), shape=shape)
    result.eliminate_zeros()

    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )
    result.eliminate_zeros()
    warnings.resetwarnings()
    sources, targets = result.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    G = ig.Graph(edges = edgelist, edge_attrs = {'weight': result.data.tolist()}) 
    return G

def build_tani_graph(clu_file, data, local = False, method = 'tani'):
    """
    readin the clu result file generated by l2knng and calculate the Tanimoto coefficients, then convert the result to an igraph.Graph as the input for the Leiden algorithm

    Parameters
    ----------
    clu_file : string
        name of the result file
    data :numpy.ndarray or scipy.sparse.csr.csr_matrix
        data matrix

    Returns
    -------
    igraph.Graph
        converted graph to be the input of the Leiden algorithm

    Raises
    ------
    RuntimeError
        result from L2knng not found
    """
    if clu_file.endswith('.clu') & os.path.isfile(clu_file):
        with open(clu_file, 'r') as f:
            clu = [line.split() for line in f.readlines()]
    elif os.path.isfile(clu_file + '.clu'):
        with open(clu_file + '.clu', 'r') as f:
            clu = [line.split() for line in f.readlines()]
    else:
        raise RuntimeError('No such a file in the current directory!')

    meta = list(map(int, clu[0]))
    csr = clu[1:]
    
    shape = (meta[0], int(meta[2] / meta[0]))
    rows = list(itertools.chain.from_iterable(itertools.repeat(i, shape[1]) for i in range(shape[0])))
    columns = list(map(int, list(itertools.chain.from_iterable(csr))[::2]))
    cos_vals = list(map(float, list(itertools.chain.from_iterable(csr))[1::2]))
    columns = list(i - 1 for i in columns) ## bug in L2knng

    if method =='tani':
        if sp.issparse(data):
            cell_norm = sp.linalg.norm(data, axis = 1)
        else:
            cell_norm = norm(data, axis = 1)
        weights = [tani_from_cos(cell_norm[i],cell_norm[j],v) for i,j,v in zip(rows, columns, cos_vals)]
    elif method =='jaccard':
        nbr_index = np.array(columns).reshape(shape)
        weights = np.ravel(map_jaccard(nbr_index)).tolist()
    elif method =='cosine':
        weights = cos_vals
    else:
        raise RuntimeError('Method not supported')

    if not local:
        G  = build_graph(rows, columns, weights, (meta[0], meta[1]))
    else:
        G  = build_graph_umap(rows, columns, weights, (meta[0], meta[1]), shape[1])
    return G

def la_find_partition(**kwargs):
    """
    Wrapper function for the Leiden algorithm
    """
    return la.find_partition(weights = 'weight',  **kwargs) 

###

def find_communities(X, pp_method = None, method = 'tani', nbr_num = 10, n_threads = None, verbose = False, keep_temp_files = False, temp_name = '', partition_type = la.RBConfigurationVertexPartition, resolution = 0.8, seed = None, local = False, **kwargs): 
    """_summary_

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse.spmatrix or pandas.DataFrame
        input data matrix
    nbr_num : int, optional
        k for kNN, by default 10
    n_threads : int, optional
        CPU threads to be used, by default 64
    verbose : bool, optional
        whether to print out intermediate alerts, by default False
    keep_temp_files : bool, optional
        whether to keep the temporal files, by default False
    temp_name : str, optional
        if keeing the temporal fiels, the file names, by default ''
    partition_type : _type_, optional
        partition type paramater to be passed to the Leiden algorithm, by default la.RBConfigurationVertexPartition
    resolution : float, optional
        resolution parameter to be passed to the Leiden algorithm, by default 0.8
    seed : _type_, optional
        random seed parameter to be passed to the Leiden algorithm, by default None

    Returns
    -------
    list
        cluster labels of each cell in the original order (rows)

    Raises
    ------
    RuntimeError
        No L2knng executable found in the directory
    RuntimeError
        System platform not supported
    """
    X = pp(X, platform = pp_method)

    tic = time.time()
    
    if os.path.isfile(temp_name+'_result.clu'):
        knng_result = temp_name+ '_result.clu'
    else:
        tic_write = time.time()
        if not(temp_name):
            uid = uuid.uuid1().hex
        else: uid = temp_name
        knng_temp = uid + '.ijv'
        if n_threads is None:
            n_threads = os.cpu_count()

        save_as_ijv(knng_temp, normalize(X, norm='l2', axis=1, copy=True))
        if verbose:
            print("#### Wrote X matrix to .ijv file in {0:.2f} seconds".format(time.time() - tic_write))

        tic_knng = time.time()
        if sys.platform.startswith("linux"):
            knng = 'p_knng_Linux'
        elif sys.platform == 'darwin':
            knng = 'knng_Mac'
        else: 
            raise RuntimeError("Operating system could not be determined or is not supported. " "sys.platform == {}".format(sys.platform), flush=True)
        core_path = os.path.dirname(__file__)
        knng_path = os.path.join(core_path, knng)
        try:
            assert os.path.isfile(knng_path)
        except AssertionError:
            print("No knng algorithm found at {}".format(knng_path), flush=True)
        knng_result = uid+ '_result.clu'
        
        if sys.platform.startswith("linux"):
                args = [knng_path, 'pl2knn',  knng_temp , '-k', str(nbr_num), '-nthreads', str(n_threads), '-readZidx', '-writeZidx',  knng_result]
        
        process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE)
        # Print realtime outputs from l2knng
        while True:
            output = process.stdout.readline()
            if process.poll() is not None:
                break
            if output:
                if verbose:
                    print(output.strip().decode('utf-8'))
        if verbose:           
            print("#### Finish find {0} exact neighbors using cosine similarity within {1:.2f} seconds".format(nbr_num, (
                    time.time() - tic_knng)))

    
    ## readin clu file and prune with Tanimoto coefficient
    tic_prune = time.time()
    G = build_tani_graph(knng_result, X, local = local, method = method)
    if verbose:
        print("#### Finish pruning process and build igraph graph in {0:.2f} seconds".format(time.time() - tic_prune))

    ## find communities using Leiden algorithm with default settings


    tic_la = time.time()
    if hasattr(partition_type, 'resolution_parameter'):
        partition = la_find_partition(graph = G, partition_type = partition_type, n_iterations = 10, resolution_parameter = resolution, seed = seed)
    else:
        partition = la_find_partition(graph = G, partition_type = partition_type, n_iterations = 10, seed = seed)
    
    communities =  partition.membership
    if verbose:
        print("#### Finish find communities using Leiden algorithm in {0:.2f} seconds".format(time.time() - tic_la))
    ## remove temp files
    
    if not(keep_temp_files):
        if verbose:
            print("#### Cleaning up", flush=True)
        for f in glob.glob(uid + '*'):
            os.remove(f)
    print("####\n#### Finish all in {0:.2f} seconds".format(time.time() - tic))
    
    return communities

## temprarly used as l2knng other than Linux is not ready
def map_tani(x, y, fcs):
    """
    map tanimoto coefficient, return the value of rows from original fcs matrix
    :param x:
    :param y:
    :param fcs:
    :return: list of tani coef
    """
    return list(map(tanimoto_coef, fcs[x], fcs[y]))

def tanimoto_coef(x, y):
    """
    calculate tanimoto coefficient
    :param x: facor, rows form fcs matrix
    :param y: facor, rows form fcs matrix
    :return: tani coef
    """
    #if sp.issparse(x):
    #    x = x.toarray()[0]
    #if sp.issparse(y):
    #    y = y.toarray()[0]     # transform sparse to dense if persent
    d = np.dot(x, y)
    return d/(np.dot(x, x) + np.dot(y, y) - d)

def calc_jaccard(i, idx):
    """
    Jaccard coefficient calculation from PhenoGraph
    :param i: index
    :param idx: full list of neighbors in the order of index from 0
    :return: jaccard coef
    """
    coefficients = np.fromiter((len(set(idx[i]).intersection(set(idx[j]))) for j in idx[i]), dtype=float)
    coefficients /= (2 * idx.shape[1] - coefficients)
    return coefficients


def map_jaccard(idx):
    """
    map jaccard coefficient
    :param idx: full list of neighbors in the order of index from 0
    :return: a list of jaccard coef values in the order of index from 0 of the original list
    """
    n = len(idx)
    with closing(Pool()) as pool:
        jaccard_values = pool.starmap(calc_jaccard, zip(range(n), repeat(idx)))
    return jaccard_values

def bc(X, nbr_num = 10, seed = None, metric = "cosine", local = False, method = 'tani', pp_method = None, **kwargs):
    X = pp(X, platform = pp_method)
    k = nbr_num
    d, idx = bruteforce_nn.knnsearch(X, k + 1, metric)
    idx = idx[:,1:]
    shape = idx.shape
    rows = list(itertools.chain.from_iterable(itertools.repeat(i, shape[1]) for i in range(shape[0])))
    columns = list(itertools.chain.from_iterable(idx))
    if method == 'tani':
        weights = map_tani(rows, columns, X)
    elif method =='jaccard':
        nbr_index = np.array(idx)
        weights = np.ravel(map_jaccard(nbr_index)).tolist()
    elif method =='cosine':
        weights = list(itertools.chain.from_iterable(d))
    else:
        raise RuntimeError('Method not supported')
    
    if not local:
        G = build_graph(rows, columns, weights, (shape[0], shape[0]))
    else:
        G  = build_graph_umap(rows, columns, weights, (shape[0], shape[0]), shape[1])
    partition = la_find_partition(graph = G, partition_type = la.RBConfigurationVertexPartition, n_iterations = 100, resolution_parameter = 0.8, seed = seed)
    communities =  partition.membership
    return(communities)


def clustering(X, nbr_num = 10, seed = None, local = False, method =  'tani', pp_method = None, **kwargs):
    if sys.platform.startswith("linux"):
        return(find_communities(X, nbr_num=nbr_num, seed = seed, local = local, method = method, pp_method = pp_method, **kwargs))
    else:
        X = pp(X)
        return(bc(X, nbr_num= nbr_num, seed = seed, local = local, method = method, pp_method = pp_method, **kwargs))
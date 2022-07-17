# CosTaL: An accurate and scalabel graph-based clustering algorithm for high-dimensional single-cell data analysis

***
## Introduction
CosTaL use p-L2knng to map k nearest neighbors and construct a kNN graph, which is later refined using Tanimoto coefficient for a better restoration of the original similarity relationships among the cells inside the high-dimensional single-cell dataset.

CosTaL is still under development in support of all platforms. Right now, the implementation of p-L2knng is only supporting the Linux system using executable, which may generate temproary files as input and output for p-L2knng (remember to get them removed if there is a problem happend during the executation, the name is the automatically-generated UUID as a 32-character lowercase hexadecimal string, unless specificed). The kNN maping on Mac and Windows platforms is temporarily using the brute force mapping borrowed from [PhenoGraph](https://github.com/dpeerlab/PhenoGraph/blob/master/phenograph/bruteforce_nn.py).


***
## Usages
#### Install
```
pip install CosTaL
```
***
#### Basic clustering (Tanimoto coefficient refined)
```
import CosTaL as ct

clusters = ct.clustering(data, method = 'mass', nbr_num = 10)
```
#### Tanimoto coefficient + [UMAP connectivity (affinity)](https://umap-learn.readthedocs.io/en/latest/index.html) doubly-refined clustering
```
clusters = ct.clustering(data, method = 'mass', nbr_num = 10, local = True)
```
***

## Authors
Yijia Li: li000678@umn.edu

## Other information

* Liscense
>This project is licensed under the [MIT] License - see the LICENSE file for details
* Developemnt status:
>Alpha

## References
[1] Anastasiu D C, Karypis G. L2knng: Fast exact k-nearest neighbor graph construction with l2-norm pruning[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. 2015: 791-800.

[2] McInnes L, Healy J, Melville J. Umap: Uniform manifold approximation and projection for dimension reduction[J]. arXiv preprint arXiv:1802.03426, 2018.

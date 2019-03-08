This folder includes five self-tuning schemes for the TPTSSR parameter M:

1- Global Supervised and 2- Global Unsupervised schemes: both schemes aim to compute one single optimal value for M that will be used in the testing stage by all test samples. The supervised method is similar to the Leave One Out Cross Validation scheme except that only training data are contributing to the estimation of M, and the unsupervised method relies on the concept of cluster boundary detection. The basic idea is to compute the parameter M using a kind of jump detection in the neighbor cluster.

3- Adaptive Supervised and 4- Adaptive Unsupervised schemes: regarding adaptive schemes, each training sample has its own optimal M value. The score is calculated in a way that focuses on a single training sample. Thus, each training sample will have its own value of M. 

5- Adaptive Semi-supervised scheme: regarding to the training stage, the adaptive supervised scheme was applied to the labeled samples, and the adaptive unsupervised scheme was applied to both labeled and unlabeled samples. 
As far as test stage, M is calculated similarly to the adaptive supervised and unsupervised scheme.


The data folder (Data) contains five image datasets in .mat format: FERET , Scene 15, Honda, UMIST, and USPS datasets.

Each dataset includes 2 fields: data matrix (mat) and label vectors (labels).
     mat: data matrix (each column represent a sample).
     labels: label vector containing the labels of mat matrix.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# For CS224u, Stanford, Spring 2015 (Chris Potts)
# Exported from distributedwordreps.ipynb, which can
# also be viewed in HTML: distributedwordreps.html

######################################################################

import os
import sys
import csv
import copy
import random
import itertools
from operator import itemgetter
from collections import defaultdict
# Make sure you've got Numpy and Scipy installed:
import numpy as np
import scipy
import scipy.spatial.distance
from numpy.linalg import svd
# For visualization:
import matplotlib.pyplot as plt
# For clustering in the 'Word-sense ambiguities' section:
from sklearn.cluster import AffinityPropagation

######################################################################
# Reading in matrices

def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):    
    reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
    colnames = None
    if header:
        colnames = reader.next()
        colnames = colnames[1: ]
    mat = []    
    rownames = []
    for line in reader:        
        rownames.append(line[0])            
        mat.append(np.array(map(float, line[1: ])))
    return (np.array(mat), rownames, colnames)

######################################################################
# Vector comparison

def euclidean(u, v):
    # Use scipy's method:
    return scipy.spatial.distance.euclidean(u, v)
    # Or define it yourself:
    # return vector_length(u - v)    

def vector_length(u):
    return np.sqrt(np.dot(u, u))

def length_norm(u):
    return u / vector_length(u)

def cosine(u, v):
    # Use scipy's method:
    return scipy.spatial.distance.cosine(u, v)
    # Or define it yourself:
    # return 1.0 - (np.dot(u, v) / (vector_length(u) * vector_length(v)))

def matching(u, v):
    # The scipy implementation is for binary vectors only. This version is more general.
    return np.sum(np.minimum(u, v))

def jaccard(u, v):
    # The scipy implementation is for binary vectors only. This version is more general.
    return 1.0 - (matching(u, v) / np.sum(np.maximum(u, v)))

def neighbors(word=None, mat=None, rownames=None, distfunc=cosine):
    if word not in rownames:
        raise ValueError('%s is not in this VSM' % word)
    w = mat[rownames.index(word)]
    dists = [(rownames[i], distfunc(w, mat[i])) for i in xrange(len(mat))]
    return sorted(dists, key=itemgetter(1), reverse=False)

######################################################################
# Reweighting

def prob_norm(u):
    return u / np.sum(u)

def pmi(mat=None, rownames=None, positive=True):
    """PMI on mat; positive=True does PPMI. rownames is not used; it's 
    an argument only for consistency with other methods used here"""
    # Joint probability table:
    p = mat / np.sum(mat, axis=None)
    # Pre-compute column sums:
    colprobs = np.sum(p, axis=0)
    # Vectorize this function so that it can be applied rowwise:
    np_pmi_log = np.vectorize((lambda x : _pmi_log(x, positive=positive)))
    p = np.array([np_pmi_log(row / (np.sum(row)*colprobs)) for row in p])   
    return (p, rownames)

def _pmi_log(x, positive=True):
    """Maps 0 and negative values to 0.0, otherwise to log.
    With positive=True, maps negative values to 0."""
    val = 0.0
    if x > 0.0:
        val = np.log(x)
    if positive:
        val = max([val,0.0])
    return val

def tfidf(mat=None, rownames=None):
    """TF-IDF on mat. rownames is unused; it's an argument only 
    for consistency with other methods used here"""
    colsums = np.sum(mat, axis=0)
    doccount = mat.shape[1]
    w = np.array([_tfidf_row_func(row, colsums, doccount) for row in mat])
    return (w, rownames)

def _tfidf_row_func(row, colsums, doccount):
    df = float(len([x for x in row if x > 0]))
    idf = 0.0
    # This ensures a defined IDF value >= 0.0:
    if df > 0.0 and df != doccount:
        idf = np.log(doccount / df)
    tfs = row/colsums
    return tfs * idf

######################################################################
# Dimensionality reduction

def lsa(mat=None, rownames=None, k=100):
    """svd with a column-wise truncation to k dimensions; rownames 
    is passed through only for consistency with other methods"""
    rowmat, singvals, colmat = svd(mat, full_matrices=False)
    singvals = np.diag(singvals)
    trunc = np.dot(rowmat[:, 0:k], singvals[0:k, 0:k])
    return (trunc, rownames)


######################################################################
# Semantic orientation method
        
def semantic_orientation(
        mat=None, 
        rownames=None,
        seeds1=['bad', 'nasty', 'poor', 'negative', 'unfortunate', 'wrong', 'inferior'],
        seeds2=['good', 'nice', 'excellent', 'positive', 'fortunate', 'correct', 'superior'],
        distfunc=cosine):
    sm1 = so_seed_matrix(seeds1, mat, rownames)
    sm2 = so_seed_matrix(seeds2, mat, rownames)
    scores = [(rownames[i], so_row_func(mat[i], sm1, sm2, distfunc)) for i in xrange(len(mat))]
    return sorted(scores, key=itemgetter(1), reverse=False)

def so_seed_matrix(seeds, mat, rownames):
    indices = [rownames.index(word) for word in seeds if word in rownames]
    if not indices:
        raise ValueError('The matrix contains no members of the seed set: %s' % ",".join(seeds))
    return mat[np.array(indices)]
    
def so_row_func(row, sm1, sm2, distfunc):
    val1 = np.sum([distfunc(row, srow) for srow in sm1])
    val2 = np.sum([distfunc(row, srow) for srow in sm2])
    return val1 - val2    

######################################################################
# Disambiguation

def disambiguate(mat=None, rownames=None, minval=0.0, mindist=20, dist_func=cosine):
    """Basic unsupervised disambiguation. minval sets what it means to occur in a column"""
    clustered = defaultdict(lambda : defaultdict(int))
    # For each word, cluster the documents containing it:
    for w_index, w in enumerate(rownames):
        doc_indices = np.array([j for j in range(mat.shape[1]) if mat[w_index,j] > minval])

        print doc_indices

        clust = cluster(mat, doc_indices) 
        for doc_index, c_index in clust:
            w_sense = "%s_%s" % (w, c_index)
            clustered[w_sense][doc_index] = mat[w_index, doc_index]
    # Build the new matrix:
    new_rownames = sorted(clustered.keys())
    new_mat = np.zeros((len(new_rownames), mat.shape[1]))
    for i, w in enumerate(new_rownames):
        for j in clustered[w]:            
            new_mat[i,j] = clustered[w][j]

    newer_rownames = sorted(clustered.keys())
    newer_mat = np.zeros((len(new_rownames), mat.shape[1]))
    combined = set();
    for i, name1 in new_rownames:
        for j, name2 in new_rownames:
            if(i >= j): continue;

            if(name1.split("_")[0] == name2.split("_")[0] and \
                    dist_func(new_mat[i], new_mat[j]) < mindist):
                newer_mat.append(new_mat[i] + new_mat[j]); #add back the vectors
                newer_rownames.append(name1.split("_")[0]); #add back the original word
                combined.add(name2);

        if(name1 not in combined): #haven't combined this word before
            newer_mat.append(new_mat[i]);


    return (new_mat, new_rownames)

def cluster(mat, doc_indices):    
    X = mat[:, doc_indices].T
    # Other clustering algorithms can easily be swapped in: 
    # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
    clust = AffinityPropagation()
    clust.fit(X)    
    return zip(doc_indices,  clust.labels_)     

######################################################################
# GloVe word representations

def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Creates an m x n matrix of random values in [lower, upper]"""
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)

def glove(
        mat=None, rownames=None, 
        n=100, xmax=100, alpha=0.75, 
        iterations=100, learning_rate=0.05, 
        display_progress=False):
    """Basic GloVe. rownames is passed through unused for compatibility
    with other methods. n sets the dimensionality of the output vectors.
    xmax and alpha controls the weighting function (see the paper, eq. (9)).
    iterations and learning_rate control the SGD training.
    display_progress=True prints iterations and current error to stdout."""    
    m = mat.shape[0]
    W = randmatrix(m, n) # Word weights.
    C = randmatrix(m, n) # Context weights.
    B = randmatrix(2, m) # Word and context biases.
    indices = range(m)
    for iteration in range(iterations):
        error = 0.0        
        random.shuffle(indices)
        for i, j in itertools.product(indices, indices):
            if mat[i,j] > 0.0:     
                # Weighting function from eq. (9)
                weight = (mat[i,j] / xmax)**alpha if mat[i,j] < xmax else 1.0
                # Cost is J' based on eq. (8) in the paper:
                diff = np.dot(W[i], C[j]) + B[0,i] + B[1,j] - np.log(mat[i,j])                
                fdiff = diff * weight                
                # Gradients:
                wgrad = fdiff * C[j]
                cgrad = fdiff * W[i]
                wbgrad = fdiff
                wcgrad = fdiff
                # Updates:
                W[i] -= (learning_rate * wgrad) 
                C[j] -= (learning_rate * cgrad) 
                B[0,i] -= (learning_rate * wbgrad) 
                B[1,j] -= (learning_rate * wcgrad)                 
                # One-half squared error term:                              
                error += 0.5 * weight * (diff**2)
        if display_progress:
            print "iteration %s: error %s" % (iteration, error)
    # Return the sum of the word and context matrices, per the advice 
    # in section 4.2:
    return (W + C, rownames)

######################################################################
# Shallow neural networks
    
from numpy import dot, outer

class ShallowNeuralNetwork:
    def __init__(self, input_dim=0, hidden_dim=0, output_dim=0, afunc=np.tanh, d_afunc=(lambda z : 1.0 - z**2)):        
        self.afunc = afunc 
        self.d_afunc = d_afunc      
        self.input = np.ones(input_dim+1)   # +1 for the bias                                         
        self.hidden = np.ones(hidden_dim+1) # +1 for the bias        
        self.output = np.ones(output_dim)        
        self.iweights = randmatrix(input_dim+1, hidden_dim)
        self.oweights = randmatrix(hidden_dim+1, output_dim)        
        self.oerr = np.zeros(output_dim+1)
        self.ierr = np.zeros(input_dim+1)
        
    def forward_propagation(self, ex):        
        self.input[ : -1] = ex # ignore the bias
        self.hidden[ : -1] = self.afunc(dot(self.input, self.iweights)) # ignore the bias
        self.output = self.afunc(dot(self.hidden, self.oweights))
        return copy.deepcopy(self.output)
        
    def backward_propagation(self, labels, alpha=0.5):
        labels = np.array(labels)       
        self.oerr = (labels-self.output) * self.d_afunc(self.output)
        herr = dot(self.oerr, self.oweights.T) * self.d_afunc(self.hidden)
        self.oweights += alpha * outer(self.hidden, self.oerr)
        self.iweights += alpha * outer(self.input, herr[:-1]) # ignore the bias
        return np.sum(0.5 * (labels-self.output)**2)

    def train(self, training_data, maxiter=5000, alpha=0.05, epsilon=1.5e-8, display_progress=False):       
        iteration = 0
        error = sys.float_info.max
        while error > epsilon and iteration < maxiter:            
            error = 0.0
            random.shuffle(training_data)
            for ex, labels in training_data:
                self.forward_propagation(ex)
                error += self.backward_propagation(labels, alpha=alpha)           
            if display_progress:
                print 'completed iteration %s; error is %s' % (iteration, error)
            iteration += 1
                    
    def predict(self, ex):
        self.forward_propagation(ex)
        return copy.deepcopy(self.output)
        
    def hidden_representation(self, ex):
        self.forward_propagation(ex)
        return self.hidden

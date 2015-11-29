#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Harley Sugarman (harleys@stanford.edu), Angelica Perez (pereza77@stanford.edu)
# CS224U, Created: 3 May 2015
# file: py

import numpy as np
from numpy.linalg import svd
import csv

global glove;

class Glove:
    
    def __init__(self, filename, delimiter = ",", header=True, quoting=csv.QUOTE_MINIMAL, v=True):
        self.v = v;
        self.filename = filename;

        self.matrix, self.rows, self.cols = self._build(filename, delimiter, header, quoting)

        self.vectors = {};
        for i, word in enumerate(self.rows):
            self.vectors[word] = self.matrix[i]

    def __str__(self):
    	return "Glove: " + self.filename + "\n    has " + str(len(self.vectors)) + " words.";

    def __repr__(self):
        return str(self);

    def __contains__(self, item):
        return item in self.vectors

    def lsa(self, k=100):
        rowmat, singvals, colmat = svd(self.matrix, full_matrices=False)
        singvals = np.diag(singvals)
        trunc = np.dot(rowmat[:, 0:k], singvals[0:k, 0:k])

        self.vectors = {};
        for i, word in enumerate(self.rows):
            self.vectors[word] = trunc[i]

    def _build(self, src_filename, delimiter, header, quoting):    
        reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
        colnames = None
        if header:
            colnames = reader.next()
            colnames = colnames[1: ]
        mat = []    
        rownames = []
        for line in reader:        
            rownames.append(line[0].lower())            
            mat.append(np.array(map(float, line[1: ])))
        return (np.array(mat), rownames, colnames)


    # Accepts a word as a parameter, Returns None if it can't find the vector. Otherwise
    def getVec(self, word):
        w = word.lower();
        return self.vectors[w] if(w in self.vectors) else None;

    # Returns a list of all the words in the glove matrix
    def getVocab(self):
        return self.vectors.keys();

    # Computes the sum of the glove vectors of all elements in words
    def getSumVec(self, words):
        targetvec = self.getVec(words[0]);
        if(targetvec == None and self.v): self.error("Glove does not have \"" + words[0] + "\" in its vocabulary");

        for word in words[1:]:
            wordvec = self.getVec(word);
            if(wordvec != None): targetvec = map(lambda i: targetvec[i] + wordvec[i], xrange(len(targetvec)));  
            else:
                if(self.v): self.error("Glove does not have \"" + word + "\" in its vocabulary");

        return targetvec

    # Computes the average of the glove vectors of all elements in words
    # If unigram model is passed in, weights each word by its inverse unigram probability
    def getAverageVec(self, words, unigrams = None):
        targetvec = [0]*len(self.vectors.values()[0])
        count = 0;
        total_sum = 0
        if unigrams != None:
            for word in words:
                wProb = 1 if unigrams == None else unigrams.getSingleNonLogScore(word)
                total_sum += float(wProb)
        for word in words:
            wordvec = self.getVec(word);
            if(wordvec != None):
                wordProb = 1 if unigrams == None else unigrams.getSingleNonLogScore(word)/total_sum
                count += 1;
                targetvec = map(lambda i: targetvec[i] + wordvec[i]*(1/wordProb), xrange(len(targetvec)));
                
            else:
                if(self.v): self.error("Glove does not have \"" + word + "\" in its vocabulary");

        if(count == 0): return None;
        return map(lambda x: x/count, targetvec);

    def error(self, msg):
        print '\033[91m' + msg + '\033[0m';

#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: Glove.py

import numpy as np
from numpy.linalg import svd
import csv

global glove;

# A class to handle and manipulate glove vectors more easily
class Glove:
    
    # filename - file to load glove vectors
    # delimiter - delimiter to use to load glove vectors
    # header - whether glove vector files has headers
    # quoting - for csv parsing
    # weighting - pmi, ppmi, tfidf or None; modifies the glove vectors
    # v -- verbosity
    def __init__(self, filename, delimiter = ",", header=True, quoting=csv.QUOTE_MINIMAL, weighting="None", v=True):
        self.v = v;
        self.filename = filename;

        self.matrix, self.rows, self.cols = self._build(filename, delimiter, header, quoting)

        # internal function for manipulating rows of glove vectors
        def _tfidf_row_func(row, colsums, doccount):
            df = float(len([x for x in row if x > 0]))
            idf = 0.0
            # This ensures a defined IDF value >= 0.0:
            if df > 0.0 and df != doccount:
                idf = np.log(doccount / df)
            tfs = row/colsums
            return tfs * idf

        # Weighting glove vectors based on tfidf, as glove vectors are based on
        # co-occurences of words
        def tfidf():
            colsums = np.sum(self.matrix, axis=0)
            doccount = self.matrix.shape[1]
            self.matrix = np.array([_tfidf_row_func(row, colsums, doccount) for row in self.matrix])

        # internal function for manipulating pmi rows
        def _pmi_log(x, positive=True):
            val = 0.0
            if x > 0.0: val = np.log(x)
            if positive: val = max([val,0.0])
            return val

        # Weighting glove vectors based on pmi, positive=True does ppmi
        def pmi(positive=True):
            # Joint probability table:
            p = self.matrix / np.sum(self.matrix, axis=None)

            # Pre-compute column sums:
            colprobs = np.sum(p, axis=0)

            # Vectorize this function so that it can be applied rowwise:
            np_pmi_log = np.vectorize((lambda x : _pmi_log(x, positive=positive)))
            self.matrix = np.array([np_pmi_log(row / (np.sum(row)*colprobs)) for row in p])   

        # Which weighting should I use? Based on weighting parameter
        if(weighting.lower() == "pmi"): pmi(positive=False);
        if(weighting.lower() == "ppmi"): pmi(positive=True);
        if(weighting.lower() == "tfidf"): tfidf();

        # Link VSM's of words to actual words themselves
        self.vectors = {};
        for i, word in enumerate(self.rows):
            self.vectors[word] = self.matrix[i]

    # For debugging purposes
    def __str__(self):
    	return "Glove: " + self.filename + "\n    has " + str(len(self.vectors)) + " words.";

    # For debugging purposes
    def __repr__(self):
        return str(self);

    # To see if a word is contained in the glove file loaded
    def __contains__(self, item):
        return item in self.vectors

    # Method to lsa the glove vectors and reduce the dimensionality
    def lsa(self, k=100):
        rowmat, singvals, colmat = svd(self.matrix, full_matrices=False)
        singvals = np.diag(singvals)
        trunc = np.dot(rowmat[:, 0:k], singvals[0:k, 0:k])

        self.vectors = {};
        for i, word in enumerate(self.rows):
            self.vectors[word] = trunc[i]

    # Loads the file in and reads it into a workable format
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
        if(targetvec is None and self.v): self.error("Glove does not have \"" + words[0] + "\" in its vocabulary");

        for word in words[1:]:
            wordvec = self.getVec(word);
            if(wordvec is not None): targetvec = map(lambda i: targetvec[i] + wordvec[i], xrange(len(targetvec)));  
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
            if(wordvec is not None):
                wordProb = 1 if unigrams == None else unigrams.getSingleNonLogScore(word)/total_sum
                count += 1;
                targetvec = map(lambda i: targetvec[i] + wordvec[i]*(1/wordProb), xrange(len(targetvec)));
                
            else:
                if(self.v): self.error("Glove does not have \"" + word + "\" in its vocabulary");

        if(count == 0): return None;
        return map(lambda x: x/count, targetvec);

    # Used for printing
    def error(self, msg):
        print '\033[91m' + msg + '\033[0m';

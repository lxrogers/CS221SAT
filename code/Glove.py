#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Harley Sugarman (harleys@stanford.edu), Angelica Perez (pereza77@stanford.edu)
# CS224U, Created: 3 May 2015
# file: py

import numpy as np
import csv

global glove;

class Glove:
    
    def __init__(self, filename, delimiter = ",", header=True, quoting=csv.QUOTE_MINIMAL, v=True):
        self.v = v;
        self.filename = filename;

        matrix, rows, self.cols = self.build(filename, delimiter, header, quoting)

        self.vectors = {};
        for i, word in enumerate(rows):
            self.vectors[word] = matrix[i]

    def __str__(self):
    	return "Glove: " + self.filename + "\n    has " + str(len(self.vectors)) + " words.";

    def __repr__(self):
        return str(self);

    def __contains__(self, item):
        return item in self.vectors

    def build(self, src_filename, delimiter, header, quoting):    
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
    def getAverageVec(self, words):
        targetvec = [0]*len(self.vectors.values()[0])
        count = 0;
        for word in words:
            wordvec = self.getVec(word);
            if(wordvec != None):
                count += 1;
                targetvec = map(lambda i: targetvec[i] + wordvec[i], xrange(len(targetvec)));
                
            else:
                if(self.v): self.error("Glove does not have \"" + word + "\" in its vocabulary");

        if(count == 0): return None;
        return map(lambda x: x/count, targetvec);

    def error(self, msg):
        print '\033[91m' + msg + '\033[0m';

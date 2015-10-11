#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Harley Sugarman (harleys@stanford.edu), Angelica Perez (pereza77@stanford.edu)
# CS224U, Created: 3 May 2015
# file: Glove.py

import numpy as np
import csv

global glove;

class Glove:
    
    def __init__(self, filename, delimiter = ",", header=True, quoting=csv.QUOTE_MINIMAL):
        
        self.filename = filename;

        matrix, rows, self.cols = self.build(filename, delimiter, header, quoting)

        self.vectors = {};
        for i, word in enumerate(rows):
            self.vectors[word] = matrix[i]

    def __str__(self):
    	string = "Glove: " + self.filename + "\n    has " + str(len(self.vectors)) + " words.";


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

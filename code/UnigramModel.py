#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: UnigramModel.py

import math, collections

# So that I can pickle unigramProbs and unigramCounts
def zero_fn():
    return 0;

# Unigram model with Laplacian smoothing
class UnigramModel:

    def __init__(self):
        self.unigramProbs = collections.defaultdict(zero_fn)
        self.unigramCounts = collections.defaultdict(zero_fn);
        self.total = 0;

    # Train unigram model
    def train(self, corpus):
        for sentence in corpus:
            for word in sentence:
                self.unigramCounts[word] = self.unigramCounts[word] + 1;
                self.total += 1;

        self.unigramCounts["<UNK>"] = 0;

        # Laplacian Smoothing
        for key in self.unigramCounts:
            self.unigramProbs[key] = float(self.unigramCounts[key] + 1)/len(self.unigramCounts);

    # Gets score of a word
    def getSingleNonLogScore(self, word):
            return self.unigramProbs[word] if word in self.unigramProbs else self.unigramProbs["<UNK>"]

    # Returns probability of the sentence naturally occuring
    def score(self, sentence):
        return sum(math.log(self.unigramProbs[word if word in self.unigramProbs else "<UNK"]) for word in sentence);
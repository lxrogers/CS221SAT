#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: BigramModel.py

import collections, math

# So that I can pickle unigramCounts and bigram Counts
def zero_fn():
    return 0;

# Laplacian-Smoothed Bigram Model
class BigramModel:

    def __init__(self):
        self.bigramCounts = collections.defaultdict(zero_fn);
        self.unigramCounts = collections.defaultdict(zero_fn);

    # Trains the model on the corpus (couns unigrams and bigrams)
    def train(self, corpus):
        for sentence in corpus:
            for i in xrange(len(sentence) - 1):  
                word1 = sentence[i];
                word2 = sentence[i + 1];
                self.bigramCounts[(word1, word2)] += 1;
                self.unigramCounts[word1] += 1;

            self.unigramCounts[sentence[len(sentence) - 1]] += 1;
        
    # Gives the log probability of the sentence happening organically based on the bigrams/unigrams
    def score(self, sentence):
        s = 0;
        for i in xrange(len(sentence) - 1):
            word1 = sentence[i];
            word2 = sentence[i+1];
            pair = (word1, word2);
            if(self.bigramCounts[pair] > 0): #have seen before
                s += math.log( float(self.bigramCounts[pair] + 1)/(self.unigramCounts[word1] + len(self.unigramCounts)));
            else:
                if(word1 in self.unigramCounts):
                    s += math.log( 1.0/(len(self.unigramCounts) + self.unigramCounts[word1]));
                else:
                    s += math.log( 1.0/len(self.unigramCounts));
        return s
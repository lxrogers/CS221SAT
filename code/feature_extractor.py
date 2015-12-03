from Glove import *
import scoring
from Question import *
import scipy
import itertools
from sklearn import svm
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from distributedwordreps import *
from os import listdir
from os.path import isfile, join
import random
import collections
import operator
import re
from BigramModel import *
from UnigramModel import *
from CustomLanguageModel import *
import numpy as np
import cPickle

# Reads a file and returns the text contents
def readFile(filename):
    with open(filename) as f: return f.read();


# Returns a list of all filenames that are recursively found down a path.
#   First param: String of initial directory to start searching
#   Second param (optional): A filter function that filters the files found. Default returns all files.
def getRecursiveFiles(path, filter_fn=lambda x: True):
    paths = [path]
    files = [];
    try:
        while(len(paths) > 0):
            path = paths[0] if paths[0][-1] != "/" else paths[0][:-1];
            children = [f for f in listdir(paths[0])];
            for child in children:
                if not isfile(join(path,child)) and "." not in f: paths.append(join(path,child));
                elif isfile(join(path,child)): files.append(join(path,child));
            paths = paths[1:]; #remove te path we just looked at
        return filter(filter_fn, files);
    except:
        error(path + " is not a directory. Exiting...", True);


def loadQuestions(directory="../data/train/"):
    files = getRecursiveFiles(directory, lambda x: x[x.rfind("/") + 1] != "." and ".txt" in x and x[-1] != '~' and "norvig" not in x.lower());
    return [Question(text) for filename in files for text in readFile(filename).split("\n\n") ];


# Model Code/Evaluation
def unigramModel(unigrams, question, target, distfunc=cosine, threshold=1, rev=False):
    return unigrams.unigramProbs(target);

def bigramModel(bigrams, question, target, distfunc=cosine, threshold=1, rev=False):
    sentence = question.getsentence()
    i = sentence.find('____')
    com1 = (sentence[i-1], sentence[i])
    com2 = (sentence[i], sentence[i+1])
    return bigrams.bigramCounts[com1] + bigrams.bigramCounts[com2]

def backOffModel(backoff, question, target, distfunc=cosine, threshold=1, rev=False):
    if(not rev):
        return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: backoff.score(x[1]))[0];
    else:
        return min([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: backoff.score(x[1]))[0];

# Sentence is an array of words
# Returns answer word by averaging the sentence passed in.
# Returns None if an answer doesn't exist in the glove vocab
# Returns -1 if no answers pass the confidence threshold
def sentenceModel(glove, question, target, distfunc=cosine, threshold=1, rev=False, unigrams=None):
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), question.getSentence()), unigrams);
    return distfunc(glove.getVec(target), targetvec)

def weightedSentenceModel(glove, question, target, distfunc=cosine, threshold=1, rev=False, unigrams):
    return sentenceModel(glove, question, target, distfunc, threshold, rev, unigrams)

def distanceModel(glove, question, distfunc=cosine, threshold=1, rev=False, answer):
    if(not rev):
        bestanswer, mindist = "", float('inf');
        return min(distfunc(glove.getVec(word), glove.getVec(answer)) for word in  filter(lambda x: x not in stopwords.words('english'), question.getSentence()))
    else:
        return 0;

def getPOSVecs(sentence):
    nounVec = []
    verbVec = []
    adjVec = []
    for word in sentence:
        synsets = wn.synsets(word)
        if len(synsets) < 1 or word in stopwords.words('english'): continue
        pos = str(synsets[0].pos())
        if pos == 'n':
            nounVec.append(word)
        elif pos == 'v':
            verbVec.append(word)
        elif pos == 'a':
            adjVec.append(word)
    return nounVec, verbVec, adjVec


def adjectiveModel(glove, question, word, distfunc=cosine, threshold=1, rev=False):
        nouns, verbs, adjectives = getPOSVecs(question.getSentence());
        if(len(adjectives) == 0): return -1
        targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), adjectives))
        if(not rev):
            return distfunc(glove.getVec(word), targetvec)
        else:
            return -1*distfunc(glove.getVec(word), targetvec)

    def verbModel(glove, question, word, distfunc=cosine, threshold=1, rev=False):
        nouns, verbs, adjectives = getPOSVecs(question.getSentence());
        targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), verbs))
        if(len(verbs) == 0): return -1
        if(not rev):
            return distfunc(glove.getVec(word), targetvec)
        else:
            return -1*distfunc(glove.getVec(word), targetvec)

    def nounModel(glove, question, word, distfunc=cosine, threshold=1, rev=False):
        nouns, verbs, adjectives = getPOSVecs(question.getSentence());
        if(len(nouns) == 0): return -1
        targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), nouns))
        if(not rev):
            return distfunc(glove.getVec(word), targetvec)
        else:
            return -1*distfunc(glove.getVec(word), targetvec)

# Create Feature Extractor for a Given Sentence
# Assumes ((Sentence, Word), Indicator) is given
# Just Uses LSA 250 for each one and threshold of 1
# Also Uses N-gram models

distances = [
    (kldist, "kldist"),
    (jsd, "jsd"),
    (cosine, "cosine"),
    (L2, "L2"),
    (jaccard, "jaccard")
];

ngram_models = [
    ("Unigram", unigramModel),
    ("Bigram", bigramModel)
];

param_models = [
    ("Sentence", sentenceModel),
    ("Distance Model", distanceModel),
    ("Adjective", adjectiveModel),
    ("Noun", nounModel),
    ("Verb", verbModel),
    ("Weighted VSM", weightedSentenceModel)
];

# TODO: add interaction variables

def createSingleExtractor(example, unigrams, bigrams, glove_none, glove_tdidf, glove_pmi, glove_pmmi):
    q = example[0]
    word = example[1]
    features = []

    # First start off with the n-gram models
    unigram_d = unigramModel(unigrams, q, word)
    bigram_d = bigramModel(bigrams, q, word)
    
    features.append(unigram_d)
    features.append(bigram_d)

    # Look at VSM models now
    for name, model in param_models:
        for d_method, d_name in distances:
            if name == "Distance Model":
                dis_none = model(glove_none, q, d_method, 1, False, word)
                dis_tdidf = model(glove_tdidf, q, d_method, 1, False, word)
                dis_pmi = model(glove_pmi, q, d_method, 1, False, word)
                dis_pmmi = model(glove_pmmi, q, d_method, 1, False, word)
                features.append(dis_none)
                features.append(dis_tdidf)
                features.append(dis_pmi)
                features.append(dis_pmmi)
            elif name == "Weighted VSM":
                dis_none = model(glove_none, q, word, d_method, 1, False, unigrams)
                dis_tdidf = model(glove_tdidf, q, word, d_method, 1, False, unigrams)
                dis_pmi = model(glove_pmi, q, word, d_method, 1, False, unigrams)
                dis_pmmi = model(glove_pmmi, q, word, d_method, 1, False, unigrams)
                features.append(dis_none)
                features.append(dis_tdidf)
                features.append(dis_pmi)
                features.append(dis_pmmi)
            else:
                dis_none = model(glove_none, q, word, d_method, 1, False)
                dis_tdidf = model(glove_tdidf, q, word, d_method, 1, False)
                dis_pmi = model(glove_pmi, q, word, d_method, 1, False)
                dis_pmmi = model(glove_pmmi, q, word, d_method, 1, False)
                features.append(dis_none)
                features.append(dis_tdidf)
                features.append(dis_pmi)
                features.append(dis_pmmi)
    
    # Add in interactions
    for i in range(len(features)-1):
        for j in range(i, len(features)-1):
            features.append(features[i]*features[j])
               
    features.append(1)
    
    return features

def createFeatureExtractorForAll(examples_file, unigrams, bigrams, glove_none, glove_tdidf, glove_pmi, glove_pmmi):
    examples = loadQuestions(directory=examples_file)
    
    glove_tdidf.lsa(250)
    glove_pmi.lsa(250)
    glove_pmmi.lsa(250)
    glove_none.lsa(250)
    
    all_features = []
    all_ys = []
    for example in examples:
        for a in example.answers:
            if(a in glove_none): continue;
            answer = 1 if a == example.correctAnswer else 0
            data = (example, a)
            all_ys.append(answer)
            features = createSingleExtractor(data, unigrams, bigrams, glove_none, glove_tdidf, glove_pmi, glove_pmmi)
            all_features.append(features)

    return (all_features, all_ys)

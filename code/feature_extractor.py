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
import math

def kldist(p,q):
    return reduce(lambda soFar,i: soFar + p[i]*np.log(p[i]/q[i]), xrange(len(p)), 0);

def jsd(p,q):
    p = map(lambda u: u/sum(p), p);
    q = map(lambda v: v/sum(q), q);
    m = .5*np.add(p,q);
    return np.sqrt(.5*kldist(p,m) + .5*kldist(q,m))

def L2(u,v):
    return reduce(lambda soFar,i: soFar + (u[i]-v[i])*(u[i]-v[i]), range(len(u)), 0);

def cosine(u, v):
    return scipy.spatial.distance.cosine(u, v)

# Model Code/Evaluation
def unigramModel(unigrams, question, target, distfunc=cosine, threshold=1, rev=False):
    return unigrams.unigramProbs[target];

def bigramModel(bigrams, question, target, distfunc=cosine, threshold=1, rev=False):
    sentence = question.getSentence()
    i = sentence.index("____")
    com1 = (sentence[i-1], sentence[i])
    com2 =  0 if i == len(sentence)-1 else (sentence[i], sentence[i+1])
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

def getStrippedAnswerWords(answer):
    answers = filter(lambda x: len(x) > 0 and x not in stopwords.words('english') + ["upon", "toward"], re.split("[ ,]", answer.lower()));
    if(len(answers) > 2):
        print "error:" + answer, answers
    assert(len(answers) <= 2) # checking to make sure correct split
    return answers if len(answers) > 0 else [answer.strip()]; # if answer is a stop word

def stripTinyWords(answer):
    space_split = re.split("[\s]", answer.lstrip())
    if len(space_split) == 2:
        return space_split[0] if len(space_split[0]) > len(space_split[1]) else space_split[1]
    elif len(space_split) == 1:
        return answer
    else:
        print "there was an error parsing answer: ", answer
        return answer

def calcVecDistance(glove, targetvec, distfunc, answer):
    vec = None;
    answer_words = getStrippedAnswerWords(answer)
    if(len(answer_words) == 1):
        # Single word answer
        vec = glove.getVec(answer_words[0]);

        # Compound answer, adding the vector
        if(any(x in answer_words[0] for x in ['\'','-'])): vec = glove.getSumVec(re.split('[\'\-]', answer_words[0]));
    else:
        # Double answer question type
        vec = glove.getAverageVec(filter(lambda y: len(y) > 0, map(lambda x: x.strip(), answer_words[0])));

    # Glove does not have the answer in its vocabulary
    if(vec == None):
        return None;
    return distfunc(vec, targetvec)



def sentenceModel(glove, question, target, distfunc=cosine, threshold=1, rev=False, unigrams=None):
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), question.getSentence()), unigrams);
    return calcVecDistance(glove, targetvec, distfunc, target)

def weightedSentenceModel(glove, question, target, unigrams, distfunc=cosine, threshold=1, rev=False):
    return sentenceModel(glove, question, target, distfunc, threshold, rev, unigrams)

def distanceModel(glove, question, answer, distfunc=cosine, threshold=1, rev=False):
    if(not rev):
        return min(calcVecDistance(glove, glove.getVec(word), distfunc, answer) for word in filter(lambda x: x not in stopwords.words('english') and x in glove, question.getSentence()))
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
    if(len(adjectives) == 0): return 2
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), adjectives))
    if targetvec is None: return 2
    return calcVecDistance(glove, targetvec, distfunc, word)

def verbModel(glove, question, word, distfunc=cosine, threshold=1, rev=False):
    nouns, verbs, adjectives = getPOSVecs(question.getSentence());
    if(len(verbs) == 0): return 2
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), verbs))
    if targetvec is None: return 2
    return calcVecDistance(glove, targetvec, distfunc, word)

def nounModel(glove, question, word, distfunc=cosine, threshold=1, rev=False):
    nouns, verbs, adjectives = getPOSVecs(question.getSentence());
    if(len(nouns) == 0): return 2
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), nouns))
    if targetvec is None: return 2
    return calcVecDistance(glove, targetvec, distfunc, word)

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

param_models = [
    ("Sentence", sentenceModel),
    ("Distance Model", distanceModel),
    ("Adjective", adjectiveModel),
    ("Noun", nounModel),
    ("Verb", verbModel),
    ("Weighted VSM", weightedSentenceModel)
];

def createSingleExtractorVSM(example, glove, unigrams):
    q = example[0]
    word = example[1]
    features = []
    # Look at VSM models now
    for name, model in param_models:
        for d_method, d_name in distances:
            dis = None
            
            if name == "Distance Model":
                dis = model(glove, q, word, d_method, 1, False)
            elif name == "Weighted VSM":
                dis = model(glove, q, word, unigrams, d_method, 1, False)
            else:
                dis = model(glove, q, word, d_method, 1, False)

            if dis == None or math.isnan(dis):
                dis = 2
            features.append(dis)
    return features


def createFeatureExtractorForAll(examples, unigrams, bigrams, glove_file):
    print "Loading Glove None"
    glove = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False);
    all_features = []
    for i in range(len(examples)*5):
        all_features.append([])
    all_ys = []
    low_ranks = [None, "pmi", "ppmi", "tfidf"];
    #low_ranks = [None]
    print "Calculating VSM Methods"
    # Get Glove Based Models
    for lr in low_ranks:
        if lr != None:
            print "Loading Glove %s" %(lr)
            glove = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, weighting=lr, v=False);
        glove.lsa(250)
        count = 0
        for example in examples:
            for a in example.answers:
                data = (example, a)
                features = createSingleExtractorVSM(data, glove, unigrams)
                all_features[count] += features
                count += 1
    print "Calculating N-Grams + Interactions"
    
    # Get answers + Unigram/Bigram + Add in interactions
    index = 0
    for example in examples:
        for i,word in enumerate(example.answers):
            if i == example.correctAnswer:
                all_ys.append(1)
            else:
                all_ys.append(0)

            unigram_d = unigramModel(unigrams, example, word)
            bigram_d = bigramModel(bigrams, example, word)
    
            all_features[index].append(unigram_d)
            all_features[index].append(bigram_d)
            
            # Bias Term
            all_features[index].append(1)
            
            #Interaction Terms
            num_feats = len(all_features[index])
            for i in range(num_feats-1):
                for j in range(i+1, num_feats-1):
                    all_features[index].append(all_features[index][i]*all_features[index][j])
            index += 1
    print "Done"
    return (all_features, all_ys)

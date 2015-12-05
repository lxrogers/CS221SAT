# Returns lists of nouns, verbs, and adjectives of sentence
# copied from Main.py
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
from main import *

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

#feature extractor for sentences
#return list with the following features:
SUPPORT_WORDS = ["moreover", "besides", "additionally", "furthermore", "in fact", "and", "therefore"]
CONTRAST_WORDS = ["although", "however", "rather than", "nevertheless", "whereas", "on the other hand", "but"]

NUM_FEATURES = 13
NOUNS_INDEX = 0
ADJECTIVES_INDEX = 1
VERBS_INDEX = 2
NEGATION_INDEX = 3
SUPPORT_INDEX = 4
CONTRAST_INDEX = 5
SEMICOLON_INDEX = 6
TOTAL_WORDS_INDEX = 7
CAPITAL_WORDS_INDEX = 8
BLANK_POSITION_INDEX = 9
COMMAS_INDEX = 10
COLON_INDEX = 11
BLANK_PERCENT_INDEX = 12 #positin of blank as percentage of total words

def extractSentenceFeatures(sentence):
	features = [0 for i in range(NUM_FEATURES)]
	POSvecs = getPOSVecs(sentence)
	#print "NOUNS:", POSvecs[0]
	#print "VERBS:", POSvecs[1]
	#print "ADJ:", POSvecs[2]
	#features[NOUNS_INDEX] = len(POSvecs[0])
	#features[VERBS_INDEX] = len(POSvecs[1])
	#features[ADJECTIVES_INDEX] = len(POSvecs[2])
	#for char in sentence:
	#	if char is ';':
	#		features[SEMICOLON_INDEX] = features[SEMICOLON_INDEX] + 1
	#	elif char is ':':
	#		features[COLON_INDEX] = features[COLON_INDEX] + 1
	#	elif char is ',':
	#		features[COMMAS_INDEX] = features[COMMAS_INDEX] + 1

	#for word in re.split("[^A-Za-z0-9_\']", sentence):
	#	features[TOTAL_WORDS_INDEX] = features[TOTAL_WORDS_INDEX] + 1
	#	if not word: continue
	#	if word[0].isupper():
	#		features[CAPITAL_WORDS_INDEX] = features[CAPITAL_WORDS_INDEX] + 1
	#	if word in SUPPORT_WORDS:
	#		features[SUPPORT_INDEX] = features[SUPPORT_INDEX] + 1
	#	elif word in CONTRAST_WORDS:
	#		features[CONTRAST_INDEX] = features[CONTRAST_INDEX] + 1

	#features[BLANK_POSITION_INDEX] = sentence.find('____')
	#features[BLANK_PERCENT_INDEX] = features[BLANK_POSITION_INDEX] * 1.0/ features[TOTAL_WORDS_INDEX]

	return features

def extractAllSentenceFeatures(questions):
    features = []
    for i, q in enumerate(questions):
        features.append(extractSentenceFeatures(q.getSentence()))
    return features

def featuresUnitTest():
	testSentence = "I want to go the Linkin park; furthermore, my ____ is quite cold; my big butt is tired"
	testFeatures = extractSentenceFeatures(testSentence)
	print testSentence
	print testFeatures
	assert testFeatures[SEMICOLON_INDEX] is 2
	assert testFeatures[BLANK_POSITION_INDEX] is 46
	assert testFeatures[COMMAS_INDEX] is 1
	assert testFeatures[SUPPORT_INDEX] is 1
	assert testFeatures[CAPITAL_WORDS_INDEX] is 2

#################################################################################
#######             MODEL EVALUATION THINGS


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
    ("Weighted VSM", weightedSentenceModel),
    ("Double Blank Combo VSM", doubleSentenceModel),
    ("Double Blank Max VSM", doubleSentenceMaxModel)
];

#low_ranks = [None, "pmi", "ppmi", "tfidf"];
low_ranks = [None]

def getModelClassifications():
    model_classes = {}
    model_classes["None"] = 0
    prev = 0
    for lr in low_ranks:
        if lr == None:
            lr = "None"
        for m_n, m_m in param_models:
            for d_m, d_n in distances:
                whole_name = lr + m_n + d_n
                model_classes[whole_name] = prev + 1
                prev += 1
    model_classes["Unigram"] = prev + 1
    model_classes["Bigram"] = prev + 2
    return model_classes

def getQuestionClassifications(questions, unigrams, bigrams, glove_file):
    model_classes = getModelClassifications()
    prelim_mapping = {} # Map of question to a list of corresponding to models that correctly predicted the answer
    # First Check if the prelim mapping is in a pickle

    if len(getRecursiveFiles("../data/ml_data/sentence_train_prelim", filter_fn=lambda a: ".pickle" in a)) > 0:
        print "found Saved Prelimninary Mappings"
        prelim_mapping = loadPickle("../data/ml_data/sentence_train_prelim/com_triandev_prelimmap.pickle")
    else:
        print "Finding Preliminary Mapping"
        # Do unigram + bigram first
        for i,question in enumerate(questions):
            u_answer = unigramModel(unigrams, question)
            b_answer = bigramModel(bigrams, question)
            if u_answer == question.getCorrectWord():
                if i in prelim_mapping:
                    prelim_mapping[i].append("Unigram")
                else:
                    prelim_mapping[i] = ["Unigram"]
            if b_answer == question.getCorrectWord():
                if i in prelim_mapping:
                    prelim_mapping[i].append("Bigram")
                else:
                    prelim_mapping[i] = ["Bigram"]

        # Do glove based now
        for lr in low_ranks:
            glove = None
            if lr == None:
                print "Loading GLove None"
                glove = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False);
                lr = "None"
            else:
                print "Loading Glove %s" %(lr)
                glove = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, weighting=lr, v=False);
            glove.lsa(25) # TODO: change to 250
            for model_name, model_form in param_models:
                for d_form, d_name in distances:
                    whole_name = lr + model_name + d_name
                    for i,q in enumerate(questions):
                        answer = None
                        if model_name == "Weighted VSM":
                            answer = model_form(glove, unigrams, q, threshold=.95)
                        else:
                            answer = model_form(glove, q, threshold=.95)
                        if answer != None and answer != -1 and answer == q.getCorrectWord():
                            if i in prelim_mapping:
                                prelim_mapping[i].append(whole_name)
                            else:
                                prelim_mapping[i] = [whole_name]
        print "saving preliminary mapping"
        savePickle(prelim_mapping, "../data/ml_data/sentence_train_prelim/com_triandev_prelimmap.pickle")
    print prelim_mapping

    # Classify each question now + return
    # For now, randomly picks out of the right ones
    real_mapping = {}
    for i,q in enumerate(questions):
        if i in prelim_mapping:
            best_model = random.choice(prelim_mapping[i])
            real_mapping[i] = model_classes[best_model]
        else:
            real_mapping[i] = model_classes["None"]
    return real_mapping

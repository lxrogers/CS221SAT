# Returns lists of nouns, verbs, and adjectives of sentence
# copied from Main.py

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import re
from Question import *

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
	POSvecs = getPOSVecs(re.split("[\s]", sentence))
	print "NOUNS:", POSvecs[0]
	print "VERBS:", POSvecs[1]
	print "ADJ:", POSvecs[2]
	features[NOUNS_INDEX] = len(POSvecs[0])
	features[VERBS_INDEX] = len(POSvecs[1])
	features[ADJECTIVES_INDEX] = len(POSvecs[2])
	for char in sentence:
		if char is ';':
			features[SEMICOLON_INDEX] = features[SEMICOLON_INDEX] + 1
		elif char is ':':
			features[COLON_INDEX] = features[COLON_INDEX] + 1
		elif char is ',':
			features[COMMAS_INDEX] = features[COMMAS_INDEX] + 1

	for word in re.split("[^A-Za-z0-9_\']", sentence):
		features[TOTAL_WORDS_INDEX] = features[TOTAL_WORDS_INDEX] + 1
		if not word: continue
		if word[0].isupper():
			features[CAPITAL_WORDS_INDEX] = features[CAPITAL_WORDS_INDEX] + 1
		if word in SUPPORT_WORDS:
			features[SUPPORT_INDEX] = features[SUPPORT_INDEX] + 1
		elif word in CONTRAST_WORDS:
			features[CONTRAST_INDEX] = features[CONTRAST_INDEX] + 1

	features[BLANK_POSITION_INDEX] = sentence.find('____')
	features[BLANK_PERCENT_INDEX] = features[BLANK_POSITION_INDEX] * 1.0/ features[TOTAL_WORDS_INDEX]

	return features

def extractAllSentenceFeatures(questions):
    features = []
    for q in questions:
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


def getModelClassifications():
    model_classification = {
        "None" : 0, 
        "Sentence kldist" : 1,
        "Sentence jsd": 2, 
        "Sentence cosine":3, 
        "Sentence L2":4,
        "Sentence jaccard":5,
        "Distance Model kldist" : 6,
        "Distance Model jsd":7, 
        "Distance Model cosine":8, 
        "Distance Model L2":9,
        "Distance Model jaccard":10,
        "Adjective kldist" : 11,
        "Adjective jsd":12, 
        "Adjective cosine":13, 
        "Adjective L2":14,
        "Adjective jaccard":15,
        "Noun kldist" : 16,
        "Noun jsd":17, 
        "Noun cosine":18, 
        "Noun L2":19,
        "Noun jaccard":20,
        "Verb kldist" : 21,
        "Verb jsd":22, 
        "Verb cosine":23, 
        "Verb L2":24,
        "Verb jaccard":25,
        "Weighted VSM kldist" : 26,
        "Weighted VSM jsd":27, 
        "Weighted VSM cosine":28, 
        "Weighted VSM L2":29,
        "Weighted VSM jaccard":30,
        "Unigram":31,
        "Bigram":32
    }
    return model_classification

def getQuestionClassifications(questions, unigrams, bigrams, glove_file):
    prelim_mapping = {} # Map of question to a list of tuples corresponding to models that correctly predicted the answer and their associated distances
    
    # Do unigram + bigram first


    # Do glove based now



    # Classify each question now + return


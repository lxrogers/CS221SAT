# Returns lists of nouns, verbs, and adjectives of sentence
# copied from Main.py

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import re

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
#return tuple with the following features:
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

	return tuple(features)

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


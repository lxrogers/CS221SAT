from Glove import *
import scoring
from Question import *
import feature_extractor
from os import listdir
from os.path import isfile, join
import random
import collections
import operator
import re
import cPickle
from BigramModel import *
from UnigramModel import *
from CustomLanguageModel import *
import sentence
import scipy
import itertools
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from distributedwordreps import *
import numpy as np
import cPickle



# Reads a json file and returns the object
def readJson(filename):
    return json.loads(readFile(filename));

def error(msg, shouldExit):
    print '\033[91m' + msg + '\033[0m';
    if(shouldExit): sys.exit();

# Loads a pickle file
def loadPickle(filename):
    return cPickle.load(file(filename));

def savePickle(obj, filename):
    w = open(filename, 'wb'); # Saving in binary format
    cPickle.dump(obj, w, -1); # Using highest protocol
    w.close();

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

# Returns (unigram_dict, bigram_dict, trigram_dict)
def getGrams(path="../data/Holmes_Training_Data/"):
    loadFile = "../data/languagemodels"
    u = UnigramModel();
    b = BigramModel();
    c = CustomLanguageModel();
    if(len(getRecursiveFiles("../data/languagemodels", filter_fn=lambda a: ".pickle" in a)) > 0 and not save):
        u.total = loadPickle("../data/languagemodels/u-total.pickle")
        u.unigramProbs = loadPickle("../data/languagemodels/u-unigramProbs.pickle")
        unigramcounts = loadPickle("../data/languagemodels/unigramCounts.pickle")
        u.unigramCounts = unigramcounts;
        b.bigramCounts = loadPickle("../data/languagemodels/b-bigramCounts.pickle")
        b.unigramCounts = unigramcounts
        #c.ngramCounts = loadPickle("../data/languagemodels/c-ngramCounts.pickle")
        #c.continuationProb = loadPickle("../data/languagemodels/c-continuationProb.pickle")
        #c.total = loadPickle("../data/languagemodels/c-total.pickle")
    else:
        files = getRecursiveFiles(path) if not isfile(path) else [path];
        for filename in files:
            sentences = readFile(filename).lower().split(".");
            sentences = map(lambda sentence: "<BEGIN> " + re.sub("[^A-Za-z\ \,\'\"]", "", sentence.replace("-"," ")).strip() + " <END>", sentences);
            sentences = map(lambda sentence: filter(lambda word: len(word) > 0, re.split("[^A-Za-z]", sentence)), sentences);

            u.train(sentences);
            b.train(sentences);
            #c.train(sentences);
        if(save):
            savePickle(u.total, "../data/languagemodels/u-total.pickle")
            savePickle(u.unigramProbs, "../data/languagemodels/u-unigramProbs.pickle")
            savePickle(u.unigramCounts, "../data/languagemodels/unigramCounts.pickle")
            savePickle(b.bigramCounts, "../data/languagemodels/b-bigramCounts.pickle")
            #savePickle(c.ngramCounts, "../data/languagemodels/c-ngramCounts.pickle")
            #savePickle(c.continuationProb, "../data/languagemodels/c-continuationProb.pickle")
            #savePickle(c.total, "../data/languagemodels/c-total.pickle")
    return u, b, c

# Training on training questions + then evaluating on Dev questions for the distance ML algorithm
def getEvaluatingTrainingData():
    global save
    save = True
    ngram_path = "../data/Holmes_Training_Data/norvig.txt"
    glove_file = "../data/glove_vectors/glove.6B.300d.txt"
    
    print "Training N-Gram Models"
    unigrams, bigrams, backoff = getGrams(path=ngram_path);
    
    print "Loading Training Questions"
    training_questions = loadQuestions(directory="../data/train/")
    
    dev_qs = []
    print "Loading Dev Questions"
    dev_qs = loadQuestions(directory="../data/dev_set/")
    
    com_questions = training_questions + dev_qs
    com_features = None
    com_labels = None
    # Check if saved
    if len(getRecursiveFiles("../data/ml_data/distance_train", filter_fn=lambda a: ".pickle" in a)) > 0:
        print "Found Saved Features"
        com_features = loadPickle("../data/ml_data/distance_train/com_traindev_features.pickle")
        com_labels = loadPickle("../data/ml_data/distance_train/com_traindev_labels.pickle")
    else:
        print "Getting AlL Features"
        com_features, com_labels = feature_extractor.createFeatureExtractorForAll(com_questions, unigrams, bigrams, glove_file)
        savePickle(com_features, "../data/ml_data/distance_train/com_traindev_features.pickle")
        savePickle(com_labels, "../data/ml_data/distance_train/com_traindev_labels.pickle")
    
    break_point = len(training_questions)*5
    training_data = (com_features[:break_point], com_labels[:break_point])
    dev_data = (com_features[break_point:], com_labels[break_point:])
    return (training_data, dev_data)

# Training on training questions and dev questions + then evaluating on Test questions for distance ML algorithm
def getTestingTrainingData():
    global save
    save = True
    ngram_path = "../data/Holmes_Training_Data/norvig.txt"
    glove_file = "../data/glove_vectors/glove.6B.50d.txt"
    
    print "Training N-Gram Models..."
    unigrams, bigrams, backoff = getGrams(path=ngram_path);
    
    print "Loading Training Questions..."
    training_questions = loadQuestions(directory="../data/train/")
    
    print "Loading Dev Questions...."
    dev_qs = loadQuestions(directory="../data/dev_set/")
    
    print "Loading Test Questions"
    test_qs = loadQuestions(directory="../data/test/")
    
    com_features = None
    com_labels = None

    com_questions = training_questions + dev_qs + test_qs
    
    # Check if testing data is already create
    if len(getRecursiveFiles("../data/ml_data/distance_test", filter_fn=lambda a: ".pickle" in a)) > 0:
        print "Found Saved Test Features"
        com_features = loadPickle("../data/ml_data/distance_test/com_traindevtest_features.pickle")
        com_labels = loadPickle("../data/ml_data/distance_test/com_traindevtest_labels.pickle")
    elif len(getRecursiveFiles("../data/ml_data/distance_train", filter_fn=lambda a: ".pickle" in a)) > 0:
        print "Found Saved Training/Dev Features"
        train_features = loadPickle("../data/ml_data/distance_train/com_traindev_features.pickle")
        train_labels = loadPickle("../data/ml_data/distance_train/com_traindev_labels.pickle")
        print "Calculating All Test Features"
        test_features, test_labels = feature_extractor.createFeatureExtractorForAll(test_qs, unigrams, bigrams, glove_file)
        com_features = train_features + test_features
        com_labels = train_labels + test_labels
        print "Saving Combined Features for later use"
        savePickle(com_features, "../data/ml_data/distance_test/com_traindevtest_features.pickle")
        savePickle(com_labels, "../data/ml_data/distance_test/com_traindevtest_labels.pickle")
        # Return Data
        training_data = (train_features, train_labels)
        testing_data = (test_features, test_labels)
        return (training_data, testing_data)
    else:
        print "No saved features found. Need to Calculate All"
        com_features, com_labels = feature_extractor.createFeatureExtractorForAll(com_questions, unigrams, bigrams, glove_file)
        print "Saving Combined Features for later use"
        savePickle(com_features, "../data/ml_data/distance_test/com_traindevtest_features.pickle")
        savePickle(com_labels, "../data/ml_data/distance_test/com_traindevtest_labels.pickle")
    
    # Separate Data + Return
    break_point = (len(training_questions) + len(dev_qs))*5
    training_data = (com_features[:break_point], com_labels[:break_point])
    test_data = (com_features[break_point:], com_labels[break_point:])
    return (training_data, test_data)

# Get training + dev set for the sentence machine learning model
def getEvaluatingTrainingSentence():
    global save
    save = True
    ngram_path = "../data/Holmes_Training_Data/norvig.txt"
    glove_file = "../data/glove_vectors/glove.6B.50d.txt" # TODO: change to 300
    
    print "Training N-Gram Models..."
    unigrams, bigrams, backoff = getGrams(path=ngram_path);
    
    print "Loading Training Questions..."
    training_questions = loadQuestions(directory="../data/train/")
    
    print "Loading Evlauation Questions..."
    dev_qs = []
    dev_qs = loadQuestions(directory="../data/dev_set/")
    
    com_questions = training_questions + dev_qs
    # Check if saved
    com_mappings = None
    if len(getRecursiveFiles("../data/ml_data/sentence_train", filter_fn=lambda a: ".pickle" in a)) > 0:
        print "Found Saved Mappings"
        com_mappings = loadPickle("../data/ml_data/sentence_train/com_traindev_mappings.pickle")
    else:
        print "Getting AlL Model-Question Mappings"
        # Get the correct classification for each model
        com_mappings = sentence.getQuestionClassifications(com_questions, unigrams, bigrams, glove_file)
        savePickle(com_mappings, "../data/ml_data/sentence_train/com_traindev_mappings.pickle")
    print com_mappings
    print "Getting Question Features"
    # Get Features for all Questions
    com_features = sentence.extractAllSentenceFeatures(com_questions)
    print com_features

    print "done."
    return #for cayman debugging purposes
    
    print len(com_questions)
    print len(com_mappings)
    print "Separating the Data"
    training_data = ([], [], [])
    dev_data = ([], [], [])
    for i,q in enumerate(com_questions):
        if q in training_questions:
            training_data[0].append(q)
            training_data[1].append(com_features[i])
            training_data[2].append(com_mappings[i])
        else:
            dev_data[0].append(q)
            dev_data[1].append(com_features[i])
            dev_data[2].append(com_mappings[i])
    print training_data
    print "HEYYYYYYYYYYYY"
    print dev_data
    return (training_data, dev_data)

def getTestingTrainingSentence():
    return True

getEvaluatingTrainingSentence()

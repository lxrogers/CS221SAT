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


# Reads a json file and returns the object
def readJson(filename):
    return json.loads(readFile(filename));

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



def getTrainingData():

    # Parameters
    global save
    save = True
    ngram_path = "../data/Holmes_Training_Data/norvig.txt"
    glove_file = "../data/glove_vectors/glove.6B.50d.txt"
    
    # Get data needed
    print "Getting Glove Vectors"
    glove_none = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False);
    glove_tfidf = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, weighting="tfidf", v=False);
    glove_pmi = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, weighting="pmi", v=False);
    glove_ppmi = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, weighting="ppmi", v=False);
    
    print "Training N-Gram Models"
    unigrams, bigrams, backoff = getGrams(path=ngram_path);
    
    print "Loading Questions"
    training_questions = loadQuestions(directory="../data/train/")
    print "Getting All Features"
    features_responses = feature_extractor.createFeatureExtractorForAll(training_questions, unigrams, bigrams, glove_none, glove_tfidf, glove_pmi, glove_ppmi)
    return features_responses


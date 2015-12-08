import scipy
import itertools
from sklearn import svm
#from nltk.tag.stanford import POSTagger
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
import scoring
from Question import *
from Glove import *
from BigramModel import *
from UnigramModel import *
from CustomLanguageModel import *
import numpy as np
import cPickle

# Loads a pickle file
def loadPickle(filename):
    return cPickle.load(file(filename));

def savePickle(obj, filename):
    w = open(filename, 'wb'); # Saving in binary format
    cPickle.dump(obj, w, -1); # Using highest protocol
    w.close();

# A command line progress bar that accepts an integer from 1-100
def update_progress(progress):
    sys.stdout.write('\r');
    sys.stdout.write('[{0}>{1}] {2}%'.format('='*int(progress/10), ' '*(10 - int(progress/10)), progress));
    sys.stdout.flush();

# Reads a file and returns the text contents
def readFile(filename):
    with open(filename, 'rb') as f: return f.read();

# Throws an error.
#   First param: String that contains error/notification
#   Second param: Whether to halt program execution or not.
def error(msg, shouldExit):
    print '\033[91m' + msg + '\033[0m';
    if(shouldExit): sys.exit();

def inform(msg):
    print '\033[93m' + str(msg) + '\033[0m';

# Prints a success (in green).
def printSuccess(message):
    print '\n\033[92m' + str(message) + '\033[0m\n';

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



# =====================================================================================================================================================
# =====================================================================================================================================================
# ================================================================= DISTANCE METRICS ==================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================

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
    return scipy.spatial.distance.cosine(u, v);

# =====================================================================================================================================================
# =====================================================================================================================================================
# ================================================================== MAIN CODE BASE ===================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================

def loadQuestions(directory="../data/train/"):
    files = []
    if(isfile(directory)):
        files = [directory];
    else:
        files = getRecursiveFiles(directory, lambda x: x[x.rfind("/") + 1] != "." and ".txt" in x and x[-1] != '~' and "norvig" not in x.lower());
    return [Question(text) for filename in files for text in readFile(filename).split("\n\n") ];

global save
save = True

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
        # c.ngramCounts = loadPickle("../data/languagemodels/c-ngramCounts.pickle")
        # c.continuationProb = loadPickle("../data/languagemodels/c-continuationProb.pickle")
        # c.total = loadPickle("../data/languagemodels/c-total.pickle")
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
            # savePickle(c.ngramCounts, "../data/languagemodels/c-ngramCounts.pickle")
            # savePickle(c.continuationProb, "../data/languagemodels/c-continuationProb.pickle")
            # savePickle(c.total, "../data/languagemodels/c-total.pickle")
    return u, b, c


def distanceSingleWords(glove, given_vector, given_answer, distfunc=cosine):
    answer_words = getStrippedAnswerWords(given_answer)
    vec = None
    if(len(answer_words) == 1):
        # Single word answer
        vec = glove.getVec(answer_words[0]);

        # Compound answer, adding the vector
        if(any(x in answer_words[0] for x in ['\'','-'])): vec = glove.getSumVec(re.split('[\'\-]', answer_words[0]));
    else:
        # Double answer question type
        vec = glove.getAverageVec(filter(lambda y: len(y) > 0, map(lambda x: x.strip(), answer_words[0])));

    # Glove does not have the answer in its vocabulary
    # i.e. we shouldn't answer the question because we don't know what an answer means
    if(vec is None):
        return None
    return distfunc(given_vector, vec)


# Given a list of possible answers (in text form), it finds the closest answer to the target vector
# 
# 
# Returns (-1, inf) if none of the answers fall within the threshold or if an answer we don't understand
# Returns (text form of answer, distance our target vector is from answer)
# If return_vec is True returns (best_vector, distance our target vector is from answer)
def findBestVector(glove, targetvec, answers, distfunc, threshold):
    ind, mindist = -1, 10e100;
    for i,answer in enumerate(answers):
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
        # i.e. we shouldn't answer the question because we don't know what an answer means
        if(vec is None):
            #if(v): error("Glove does not have the means to evaluate \"" + answer + "\" in its vocabulary", False);
            return (-1, float('inf'));

        if( distfunc(vec, targetvec) < min(mindist, threshold) ):
            ind, mindist = i, distfunc(vec, targetvec);

    if (ind == -1):
        return (-1, mindist)

    return (answers[ind], mindist);

# Given a question's text answers, returns a list of words (either 1 or 2) that the answer is
def getStrippedAnswerWords(answer):
    answers = filter(lambda x: len(x) > 0 and x not in stopwords.words('english') + ["upon", "toward"], re.split("[ ,]", answer.lower()));
    if(len(answers) > 2):
        print "error:" + answer, answers
    assert(len(answers) <= 2) # checking to make sure correct split
    return answers if len(answers) > 0 else [answer.strip()]; # if answer is a stop word


# Get Clue Words Between Blanks in Double Blank Questions
# if more support words, return "support", if more contrast words, return "contrast"
def getDoubleBlankEliminationMode(question):
    #between_text_words = re.compile('____(.*?)___').search(question.text)
    if len(re.findall ( '____(.*?)____', question.text, re.DOTALL)) == 0:
        print "DOUBLE BLANK ERROR ON:"
        print question
        return "neutral"
    between_text_words = re.findall ( '____(.*?)____', question.text, re.DOTALL)[0]
    support_words = ["moreover", "besides", "additionally", "furthermore", "in fact", "and", "therefore"]
    contrast_words = ["although", "however", "rather than", "nevertheless", "whereas", "on the other hand", "but"]
    
    support, contrast = 0, 0
    for support_word in support_words:
        if support_word in between_text_words: support += 1
    for contrast_word in contrast_words:
        if contrast_word in between_text_words: contrast += 1
    if support > contrast: return "support"
    if contrast > support: return "contrast"
    else: return "neutral"

#given mode = "support", answers that are too disimilar will be eliminated and vice-versa for "contrast"
#threshold ex.: threshold = .3
#support mode: answers with distance > .7 will be eliminated; contrast mode: answers with distance < .3 will be eliminated
def getRemainingAnswers(glove, mode, question, distfunc=cosine, threshold=.6):
    if mode == "neutral":
        return question

    remaining_answers = []
    for answer in question.answers:
        answer_list = getStrippedAnswerWords(answer)
        vec1, vec2 = glove.getVec(answer_list[0]), glove.getVec(answer_list[1])
        if vec1 is None or vec2 is None:
            remaining_answers.append(answer)
            continue
        dist = distfunc(vec1, vec2)
        if (mode == "support" and dist > (1-threshold)) or (mode == "contrast" and dist < threshold):
            continue
        else:
            remaining_answers.append(answer)
    question2 = copy.copy(question)
    question2.answers = remaining_answers
    return question2

# Find the distane between two words in the same answer
def answerWordDistance(glove, answer, distfunc=cosine):
    answer_list = getStrippedAnswerWords(answer)
    vec1, vec2 = glove.getVec(answer_list[0]), glove.getVec(answer_list[1])
    if vec1 is None or vec2 is None:
        return .5
    return distfunc(vec1, vec2)

def getMaxDoubleBlankAnswer(glove, mode, question, distfunc=cosine):
    distances = [(answer, answerWordDistance(glove, answer)) for answer in question.answers]
    if mode == "support":
        return min(distances, key=lambda x: x[1])
    else:
        return max(distances, key=lambda x: x[1])

# Returns lists of nouns, verbs, and adjectives of sentence
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

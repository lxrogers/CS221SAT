
#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: main.py

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



# =====================================================================================================================================================
# =====================================================================================================================================================
# ================================================================= UTILITY FUNCTIONS =================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================


# A command line progress bar that accepts an integer from 1-100
def update_progress(progress):
    sys.stdout.write('\r');
    sys.stdout.write('[{0}>{1}] {2}%'.format('='*int(progress/10), ' '*(10 - int(progress/10)), progress));
    sys.stdout.flush();

# Splits an array into numChunks
def chunk(array, numChunks):
    for i in xrange(0, len(array), len(array)/numChunks):
        yield array[i:min(i+len(array)/numChunks, len(array))]

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

# Writes a matrix to a file with an optional delimiter
def matrixToFile(matrix, filename, delimiter=','):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerows(matrix)

# Auto-detects delimiter if no delimiter is given and converts a file (e.g. csv, tsv) into a matrix
def fileToMatrix(filename, delimiter=None):
  matrix = readFile(filename).split("\n");
  if(delimiter == None): delimiter = csv.Sniffer().sniff(matrix[1] if len(matrix) >= 1 else matrix[0]).delimter
  return map(lambda x: x.split(delimiter) , matrix);

# Filters a matrix by column by the column_index; equivalent to a WHERE clause
def filterByCol(matrix, column_index, filter_fn):
  return filter(lambda x: filter_fn(x[column_index]), matrix);

# Maps a matrix column at the column_index
def mapByCol(matrix, column_index, map_fn):
  return filter(lambda x: map_fn(x[column_index]), matrix);

# Outputs array for visualization in mathematica
def mathematicatize(array):
    return str(array).replace("[","{").replace("]","}").replace("(","{").replace(")","}");

# A command line progress bar that accepts an integer from 1-100
def update_progress(progress):
    sys.stdout.write('\r');
    sys.stdout.write('[{0}>{1}] {2}%'.format('='*int(progress/10), ' '*(10 - int(progress/10)), progress));
    sys.stdout.flush();

# Reads a file and returns the text contents
def readFile(filename):
    with open(filename) as f: return f.read();

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

# Finds the best answer given a target vector, answers, a distance function and a threshold
# Returns -1 if none of the answers fall within the threshold
# Returns None if an answer has a word we don't understand (the question is illegible);
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
        if(vec == None):
            #if(v): error("Glove does not have the means to evaluate \"" + answer + "\" in its vocabulary", False);
            return None;
        if( distfunc(vec, targetvec) < mindist and distfunc(vec, targetvec) < threshold ):
            ind, mindist = i, distfunc(vec, targetvec);
    if (ind == -1):
        return -1
    return answers[ind];

#return a list of words stripped of irrelevant words
#cleaving to -> [cleaving]
#cleaving to, ineffable -> [cleaving, ineffable]
def getStrippedAnswerWords(answer):
    comma_split = re.split("[\,]", answer)
    if len(comma_split) == 2: #double blank
        return [stripTinyWords(comma_split[0]), stripTinyWords(comma_split[1])]
    elif len(comma_split) == 1: #single blank
        return [stripTinyWords(answer)]
    else:
        print "there was an error parsing answer: ", answer
        return answer

def stripTinyWords(answer):
    space_split = re.split("[\s]", answer.lstrip())
    if len(space_split) == 2:
        return space_split[0] if len(space_split[0]) > len(space_split[1]) else space_split[1]
    elif len(space_split) == 1:
        return answer
    else:
        print "there was an error parsing answer: ", answer
        return answer

# Get Clue Words Between Blanks in Double Blank Questions
# if more support words, return "support", if more contrast words, return "contrast"
def getDoubleBlankEliminationMode(question):
    #between_text_words = re.compile('____(.*?)___').search(question.text)
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

def answerWordDistance(glove, answer, distfunc=cosine):
    answer_list = getStrippedAnswerWords(answer)
    vec1, vec2 = glove.getVec(answer_list[0]), glove.getVec(answer_list[1])
    if vec1 is None or vec2 is None:
        return .5
    return distfunc(vec1, vec2)

def getMaxDoubleBlankAnswer(glove, mode, question, distfunc=cosine):
    distances = [(answer, answerWordDistance(glove, answer)) for answer in question.answers]
    if mode == "support":
        return min(distances, key=lambda x: x[1])[0]
    else:
        return max(distances, key=lambda x: x[1])[0]

# Gets Synonyms
def getSynonyms(word):
    return list(set(synset.name()[:-10] for synset in wn.synsets(word)))


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

#####################################################################################################################
################################################### MODELS ##########################################################
#####################################################################################################################


def adjectiveModel(glove, question, distfunc=cosine, threshold=2, rev=False):
    nouns, verbs, adjectives = getPOSVecs(question.getSentence());
    if(len(adjectives) == 0): return -1
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), adjectives))
    if(not rev):
        return findBestVector(glove, targetvec, question.answers, distfunc, threshold);
    else:
        return findBestVector(glove, targetvec, question.answers, lambda x,y: -1*distfunc(x,y), threshold)

def verbModel(glove, question, distfunc=cosine, threshold=2, rev=False):
    nouns, verbs, adjectives = getPOSVecs(question.getSentence());
    if(len(verbs) == 0): return -1
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), verbs))
    if(not rev):
        return findBestVector(glove, targetvec, question.answers, distfunc, threshold);
    else:
        return findBestVector(glove, targetvec, question.answers, lambda x,y: -1*distfunc(x,y), threshold)

def nounModel(glove, question, distfunc=cosine, threshold=2, rev=False):
    nouns, verbs, adjectives = getPOSVecs(question.getSentence());
    if(len(nouns) == 0): return -1
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), nouns))
    if(not rev):
        return findBestVector(glove, targetvec, question.answers, distfunc, threshold);
    else:
        return findBestVector(glove, targetvec, question.answers, lambda x,y: -1*distfunc(x,y), threshold)

    # Sentence is an array of words
    # Returns answer word by averaging the sentence passed in.
    # Returns None if an answer doesn't exist in the glove vocab
    # Returns -1 if no answers pass the confidence threshold
def sentenceModel(glove, question, distfunc=cosine, threshold=2, rev=False, unigrams=None):
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), question.getSentence()), unigrams);
    if(not rev):
        return findBestVector(glove, targetvec, question.answers, distfunc, threshold);
    else:
        return findBestVector(glove, targetvec, question.answers, lambda x,y: -1*distfunc(x,y), threshold)

def weightedSentenceModel(glove, unigrams, question, distfunc=cosine, threshold=2, rev=False):
    return sentenceModel(glove, question, distfunc, threshold, rev, unigrams)

def doubleSentenceModel(glove, question, distfunc=cosine, threshold=2, rev=False):
    answer_words = getStrippedAnswerWords(question.answers[0])
    if(len(answer_words) == 1):
        #single blank answer
        return sentenceModel(glove, question, distfunc, threshold, rev)
    elif(len(answer_words) == 2):    
        #double blank answer
        elimination_mode = getDoubleBlankEliminationMode(question) #step 1: use clue words to determine which answers to eliminate (similar or different)
        question2 = getRemainingAnswers(glove, elimination_mode, question) #step 2: eliminate those words
        return sentenceModel(glove, question2, distfunc, threshold, rev) #step 3: find best answer out of un-eliminated words

def doubleSentenceMaxModel(glove, question, distfunc=cosine, threshold=2, rev=False):
    answer_words = getStrippedAnswerWords(question.answers[0])
    if(len(answer_words) == 1):
        #single blank answer
        return sentenceModel(glove, question, distfunc, threshold, rev)
    elif(len(answer_words) == 2):    
        #double blank answer
        elimination_mode = getDoubleBlankEliminationMode(question) #step 1: use clue words to determine which answers to eliminate (similar or different)
        if elimination_mode == "neutral":
            return sentenceModel(glove, question, distfunc, threshold, rev)
        else:
            return getMaxDoubleBlankAnswer(glove, elimination_mode, question)
            

def distanceModel(glove, question, distfunc=cosine, threshold=2, rev=False):
    if(not rev):
        bestanswer, mindist = "", float('inf');

        for answer, word in itertools.product(question.answers, filter(lambda x: x not in stopwords.words('english'), question.getSentence())):
            if(answer not in glove or word not in glove): continue;
            dist = distfunc(glove.getVec(answer), glove.getVec(word));
            if(dist < mindist):
                mindist, bestanswer = dist,answer
        return bestanswer

    else:
        return 0;

def unigramModel(unigrams, question, distfunc=cosine, threshold=2, rev=False):
    if(not rev):
        return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: unigrams.score(x[1]))[0];
    else:
        return min([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: unigrams.score(x[1]))[0];


def bigramModel(bigrams, question, distfunc=cosine, threshold=2, rev=False):
    if(not rev):
        return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: bigrams.score(x[1]))[0];
    else:
        return min([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: bigrams.score(x[1]))[0];


def backOffModel(question, distfunc=cosine, threshold=2, rev=False):
    if(not rev):
        return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: backoff.score(x[1]))[0];
    else:
        return min([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: backoff.score(x[1]))[0];



# Main method
# Global variables: unigrams, bigrams, backoff, glove, tagger
def main(questions, glove):

    #####################################################################################################################
    ################################################### MODELS ##########################################################
    #####################################################################################################################

    # Returns answer word based on random chance, given the answers 
    def randomModel(question, distfunc=cosine, threshold=2, rev=False):
        return question.answers[random.randint(0,len(question.answers)) - 1];

    #####################################################################################################################
    ################################################# EVAL MODELS #######################################################
    #####################################################################################################################

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
        ("Weighted VSM", weightedSentenceModel),
        ("Double Blank Combo VSM", doubleSentenceModel),
        ("Double Blank Max VSM", doubleSentenceMaxModel),
        ("Adjective Model", adjectiveModel),
        ("Noun Model", nounModel),
        ("Verb Model", verbModel)
    ];

    for name, model in param_models:
        if name == "Weighted VSM":
            scoring.score_model( [(model(glove, unigrams, q, threshold=.9), q.getCorrectWord()) for q in questions], verbose=True, modelname=name)
        else:
            scoring.score_model( [(model(glove, q, threshold=.9), q.getCorrectWord()) for q in questions], verbose=True, modelname=name)

    os.system("say Finished");

# =====================================================================================================================================================
# =====================================================================================================================================================
# =============================================================== COMMAND LINE REFERENCE ==============================================================
# =====================================================================================================================================================
# =====================================================================================================================================================

# Command Line Reference:
#   Example call: python main.py  -v -g ../data/glove_vectors/glove.6B.300d.txt -f ../data/Holmes_Training_Data/norvig.txt -train
#   1) -v: if you want this program to be verbose
#   2) -g: filename for glove vectors, default "../data/glove_vectors/glove.6B.50d.txt"
#   3) -train/-test: designates whether you want to evaluate on train or test (required)
#   4) -f: folder or file to read text from, default "../data/Holmes_Training_Data/"
#   5) -save: retrains the models and saves the trained models for further runs
if __name__ == "__main__":

    # Preliminary loading to get arguments
    import sys
    import time

    args = sys.argv[1:];
    start = time.time();

    v = reduce(lambda a,d: a or d== "-v", args, False);
    if(v): inform("\nImporting modules...")

    g = "../data/glove_vectors/glove.6B.50d.txt";
    f = "../data/Holmes_Training_Data/"
    train = None;
    save = False;

    # Get command lime arguments
    for i, arg in enumerate(args):
        if(arg == "-g"):
            g = args[i+1];
        if(arg == "-f"):
            f = args[i+1];
        if(arg == "-train"):
            train = True;
        if(arg == "-test"):
            train = False;
        if(arg == "-save"):
            save = True;

    # Report error if called the wrong way
    if("help" in args or train == None):
        error("Example call: python main.py -train -v -g ../data/glove_vectors/glove.6B.300d.txt\n" + 
            "   1) -v: if you want this program to be verbose\n" +
            "   2) -g: path to glove vector file (defaults to '../data/glove_vectors/glove.6B.50d.txt'\n" + 
            "   3) -train/-test: designates whether you want to evaluate on train or test (required)\n" + 
            "   4) -f: files to read text from, default '../data/Holmes_Training_Data/'\n" + 
            "   5) -save: will save the language models so you don't have to train them", True)


    # Loading Modules
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
    if(v): print "All modules successfully loaded in " + str(int(time.time() - start)) +  " seconds!"


    # Initialize all the external data
    if(v): print "Loading all external data...";
    t = time.time();

    if(v): print "\tLoading passages...";
    questions = loadQuestions(directory="../data/dev_set/") if train else loadQuestions(directory="../data/test/");
    
    # Initialize global variables
    global backoff
    global tagger

    if(v):
        if(save): print "\tTraining Language Models...";
        else: print "\tLoading Language Models...";
    unigrams, bigrams, backoff = getGrams(path=f);

    if(v): print "\tLoading Glove Vectors...";
    glove = Glove(g, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False);

    if(v): print "\tInitializing Part-Of-Speech Classifier";
    #tagger = POSTagger(
    #        'stanford-postagger/models/english-bidirectional-distsim.tagger', 
    #        'stanford-postagger/stanford-postagger.jar',
    #        'utf-8'
    #    );

    if(v): print "Finished loading all external data in " + str(int(time.time() - start)) + " seconds!"
    if(v): print "Starting program now..."

    # Main Method
    main(questions, glove);

    # Finished main execution
    if(v): printSuccess("Program successfully finished and exited in " + str(int(time.time() - start)) +  " seconds!");
    sys.exit();

# =====================================================================================================================================================
# =====================================================================================================================================================
# =================================================================== EXAMPLE CALLS ===================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================


"""
Example call of POSTagger:
======================================================
tagger = POSTagger(
        'stanford-postagger/models/english-bidirectional-distsim.tagger', 
        'stanford-postagger/stanford-postagger.jar',
        'utf-8'
    );

tagger.tag_sents(array_of_string_sentences);


Example call of  Questions
======================================================
Question(text) <= constructor, read in with loadQuestions
Question.text = "question prompt text"
question.answers = ["answer #0", "answer #1"]
question.correctAnswer = int_of_correct_answer <= corresponds with index of answers

Example call of Glove Module
======================================================
glove = Glove(filename);
print glove.getVec("and"); # <= prints out glove vector for that word, or None if word not in vocab
if("and" in glove): print glove.getVec("and"); # <= solution to see if word is in the vocab
print glove.getVocab(); # <= returns an array of all the words the glove vectors have
"""







#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: main.py


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
    return scipy.spatial.distance.cosine(u, v)

def L1(u,v):
    return distributedwordreps.L1(u,v)

def jaccard(u,v):
    return distributedwordreps.jaccard(u,v);


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

# Finds the best answer given a target vector, answers, a distance function and a threshold
# Returns -1 if none of the answers fall within the threshold
# Returns None if an answer has a word we don't understand (the question is illegible);
def findBestVector(targetvec, answers, distfunc, threshold):
    ind, mindist = -1, 10e100;
    for i,answer in enumerate(answers):
        vec = None;
        if(len(re.split("[\,\s]", answer)) <= 1):
            # Single word answer
            vec = glove.getVec(answer);

            # Compound answer, adding the vector
            if(any(x in answer for x in ['\'','-'])): vec = glove.getSumVec(re.split('[\'\-]', answer));
        else:
            # Double answer question type
            vec = glove.getAverageVec(filter(lambda y: len(y) > 0, map(lambda x: x.strip(), re.split("[\,\s]", answer))));

        # Glove does not have the answer in its vocabulary
        if(vec == None):
            if(v): error("Glove does not have the means to evaluate \"" + answer + "\" in its vocabulary", False);
            return None;

        if( distfunc(vec, targetvec) < mindist and distfunc(vec, targetvec) < threshold ):
            ind, mindist = i, distfunc(vec, targetvec);

    return answers[ind];

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




# Main method
# Global variables: unigrams, bigrams, backoff, glove, tagger
def main(questions):

    #####################################################################################################################
    ################################################### MODELS ##########################################################
    #####################################################################################################################

    # Returns answer word based on random chance, given the answers 
    def randomModel(question, distfunc=cosine, threshold=1, rev=False):
        return question.answers[random.randint(0,len(question.answers)) - 1];

    # Sentence is an array of words
    # Returns answer word by averaging the sentence passed in.
    # Returns None if an answer doesn't exist in the glove vocab
    # Returns -1 if no answers pass the confidence threshold
    def sentenceModel(question, distfunc=cosine, threshold=1, rev=False, unigrams=None):
        targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), question.getSentence()), unigrams);
        if(not rev):
            return findBestVector(targetvec, question.answers, distfunc, threshold);
        else:
            return findBestVector(targetvec, question.answers, lambda x,y: -1*distfunc(x,y), threshold)

    def weightedSentenceModel(question, distfunc=cosine, threshold=1, rev=False):
        return sentenceModel(question, distfunc, threshold, rev, unigrams)

    def distanceModel(question, distfunc=cosine, threshold=1, rev=False):
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


    def unigramModel(question, distfunc=cosine, threshold=1, rev=False):
        if(not rev):
            return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: unigrams.score(x[1]))[0];
        else:
            return min([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: unigrams.score(x[1]))[0];


    def bigramModel(question, distfunc=cosine, threshold=1, rev=False):
        if(not rev):
            return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: bigrams.score(x[1]))[0];
        else:
            return min([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: bigrams.score(x[1]))[0];


    def backOffModel(question, distfunc=cosine, threshold=1, rev=False):
        if(not rev):
            return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: backoff.score(x[1]))[0];
        else:
            return min([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: backoff.score(x[1]))[0];


    def neuralNetModel(question, distfunc=cosine, threshold=1, rev=False):

        targetvec = net.predict(getNetInput(question.getSentence()));

        if(not rev):
            return findBestVector(targetvec, question.answers, distfunc, threshold);
        else:
            return findBestVector(targetvec, question.answers, lambda x,y: -1*distfunc(x,y), threshold)
       

    #####################################################################################################################
    ############################################### MAIN CODEBASE #######################################################
    #####################################################################################################################


    def getNetInput(sentence):
        # Filtering stop words from the sentence
        sentence = filter(lambda x: x in stopwords.words('english') and (x in glove or '_' in x), sentence);

        # Add 10 or less non-stop words leading up to blank
        vsmInput = [];
        for i in xrange(10):
            if(i >= len(sentence) or '_' in sentence[i]): break;
            vsmInput += glove.getVec(sentence[i]).tolist();

        # Append 0's to make uniform input length
        if(len(vsmInput) < 500): vsmInput += [0]*(500 - len(vsmInput));
        return vsmInput;
     
    def trainNeuralNetwork(questionData):
        if(len(getRecursiveFiles("../data/neuralnet", filter_fn=lambda a: ".pickle" in a)) > 0 and not save):
            net.input = loadPickle("../data/neuralnet/input.pickle");
            net.hidden = loadPickle("../data/neuralnet/hidden.pickle");
            net.output = loadPickle("../data/neuralnet/output.pickle");
            net.iweights = loadPickle("../data/neuralnet/iweights.pickle");
            net.oweights = loadPickle("../data/neuralnet/oweights.pickle");
            net.oerr = loadPickle("../data/neuralnet/oerr.pickle");
            net.ierr = loadPickle("../data/neuralnet/ierr.pickle");
        else:
            # Creating training data
            train = [];
            for q in questionData:
                if(q.getCorrectWord() in glove):
                    train.append( (getNetInput(q.getSentence()), glove.getVec(q.getCorrectWord())) );

            if(save):
                savePickle(net.input, "../data/neuralnet/input.pickle");
                savePickle(net.hidden, "../data/neuralnet/hidden.pickle");
                savePickle(net.output, "../data/neuralnet/output.pickle");
                savePickle(net.iweights, "../data/neuralnet/iweights.pickle");
                savePickle(net.oweights, "../data/neuralnet/oweights.pickle");
                savePickle(net.oerr, "../data/neuralnet/oerr.pickle");
                savePickle(net.ierr, "../data/neuralnet/ierr.pickle");

        net.train(train); 

    # Initializing Shallow Neural Network using 50 length VSMs and 10 words, using sigmoid layers
    # sigmoid = np.vectorize(lambda x: 1.0/(1.0+np.exp(-x)))
    # net = ShallowNeuralNetwork(
    #     input_dim=500,
    #     hidden_dim=75,
    #     output_dim=50,
    #     afunc= sigmoid, # Sigmoid Layer
    #     d_afunc=np.vectorize(lambda x: sigmoid(x)*(1-sigmoid(x)))
    # )

    # trainNeuralNetwork(questions[:len(questions)/2]);


    #####################################################################################################################
    ################################################# EVAL MODELS #######################################################
    #####################################################################################################################

    models = [
        ("Random", randomModel),
        ("Sentence", sentenceModel),
        ("Unigram", unigramModel),
        ("Bigram", bigramModel),
        ("Distance Model", distanceModel),
        ("Weighted VSM", weightedSentenceModel)
        #("Neural Network", neuralNetModel)
        #("BackOff", backOffModel) 
    ];

    distances = [
        (kldist, "kldist"),
        (jsd, "jsd"),
        (cosine, "cosine"),
        (L2, "L2"),
        (L1, "L1"),
        (jaccard, "jaccard")
    ];

    for name, model in models:
            scoring.score_model( [(model(q), q.getCorrectWord()) for q in questions], verbose=True, modelname=name)

    for name, model in models:
            scoring.score_elimination_model( [(model(q, rev=True), q.getCorrectWord()) for q in questions], verbose=True, modelname=name)
    
    # Look at threshold models
    test_thresholds = [1, .9, .98, .99, .95, .8, .85, .01, .1, .2, .5, .001]
    for test_t in test_thresholds:
        name_reg = "Sentence Model - Threshold " + str(test_t)
        scoring.score_model( [(sentenceModel(q, cosine, test_t, False, None), q.getCorrectWord()) for q in questions], verbose=True, modelname=name_reg)
        name_weighted = "Weighted VSM - Threshold " + str(test_t)
        scoring.score_model( [(weightedSentenceModel(q, cosine, test_t, False), q.getCorrectWord()) for q in questions], verbose=True, modelname=name_weighted)


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
    global unigrams
    global bigrams
    global backoff
    global glove
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
    main(questions);

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






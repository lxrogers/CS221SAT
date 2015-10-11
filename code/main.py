
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
    return pickle.load(file(filename));

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


# Allows for folding data for testing purposes. Usage Example:
# ======================================================================
# folds = FoldData(trainingdata, labels, folds=5);
# for train, test in folds:
#     classifier.train(train['data'], train['labels']);
#     classifier.evaluate(test['data'], test['labels']);

class FoldData:
    data = [];
    labels = [];
    folds = 10;
    current = 0;
    
    def __init__(self, data, labels, folds=10):
        # TODO: assert that data and labels are same length
        
        self.folds = folds;
        i = np.arange(len(data));
        np.random.shuffle(i);
                
        seen = set();
        for indices in chunk(i, self.folds):
            indices = list(set(indices).difference(seen));
            
            self.data.append([data[index] for index in indices]);
            self.labels.append([labels[index] for index in indices]);
            
            seen.update(indices);
        
        setattr(self, "folds", self.folds);
    
    def __iter__(self):
        return self;

    def next(self):
        if self.current >= self.folds:
            raise StopIteration
        else:
            self.current += 1
            return self.getFold(self.current - 1)
        
    def getFold(self, index):
        train = {'data':[],'labels':[]};
        test  = {'data':[],'labels':[]};
        for fold in xrange(self.folds):
            if(fold != index):
                train['data'] += self.data[fold];
                train['labels'] += self.labels[fold];
            else:
                test['data'] += self.data[fold];
                test['labels'] += self.labels[fold];

        return (train, test);


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

def loadQuestions(qfile="../data/train/questions-train.txt", afile="../data/train/answers-train.txt"):
    qtext, atext = readFile(qfile), readFile(afile);
    qtext = filter(lambda y: len(y) > 0, map(lambda x: x.strip(), re.split('\n\n', qtext)));
    atext = filter(lambda y: len(y) > 0, re.split('\n', atext));

    corrects = dict((a[:a.find(')')], a[a.rfind(' '):].strip()) for a in atext);
    questions = [];
    for q in qtext:
        arr = q.split("\n");
        number = arr[0][:arr[0].find(')')];
        prompt = arr[0][arr[0].find(' '):];
        answers = map(lambda x: x[x.find(')')+1:].strip(), arr[1:]);
        questions.append(Question(prompt, number, answers, answers.index(corrects[number])));

    return questions


# Computes the sum of the glove vectors of all elements in words
def getSumVec(words, glove):
    targetvec = glove.getVec(words[0]);
    if(targetvec == None and v): error("Glove does not have \"" + words[0] + "\" in its vocabulary", False);

    for word in words[1:]:
        wordvec = glove.getVec(word);
        if(wordvec != None): targetvec = map(lambda i: targetvec[i] + wordvec[i], xrange(len(targetvec)));  
        else:
            if(v): error("Glove does not have \"" + word + "\" in its vocabulary", False);

    return targetvec

# Computes the average of the glove vectors of all elements in words
def getAverageVec(words, glove):
    start = 0;
    targetvec = glove.getVec(words[start]);
    while(targetvec == None):
        if(v): error("Glove does not have \"" + words[start] + "\" in its vocabulary", False);
        start += 1;
        targetvec = glove.getVec(words[start]);

    count = 0;
    for word in words[start:]:
        wordvec = glove.getVec(word);
        if(wordvec != None):
            count += 1;
            targetvec = map(lambda i: targetvec[i] + wordvec[i], xrange(len(targetvec)));
            
        else:
            if(v): error("Glove does not have \"" + word + "\" in its vocabulary", False);

    return map(lambda x: x/count, targetvec);

# Returns (unigram_dict, bigram_dict, trigram_dict)
def getGrams(path="../data/Holmes_Training_Data/"):
    unigramCounts = collections.defaultdict(lambda: 1);
    bigramCounts = collections.defaultdict(lambda: []);
    trigramCounts = collections.defaultdict(lambda: []);

    files = getRecursiveFiles(path) if not isfile(path) else [path];

    for filename in files:
        sentences = readFile(filename).lower().split(".");
        sentences = map(lambda sentence: re.sub("[^A-Za-z\ \,\'\"]", "", sentence.replace("-"," ")).strip(), sentences);
        sentences = map(lambda sentence: filter(lambda word: len(word) > 0, re.split("[^A-Za-z]", sentence)), sentences);

        for sentence in sentences:
            for i, word in enumerate(sentence):
                unigramCounts[word] += 1;
                if(i + 1 < len(sentence)): bigramCounts[word] += [sentence[i+1]]
                if(i + 2 < len(sentence)): trigramCounts[(word, sentence[i+1])] += [sentence[i+2]];

    return unigramCounts, bigramCounts, trigramCounts

# Finds the best answer given a target vector, answers, a distance function and a threshold
# Returns -1 if none of the answers fall within the threshold
# Returns None if an answer has a word we don't understand (the question is illegible);
def findBestVector(targetvec, answers, glove, distfunc, threshold):
    ind, mindist = -1, 10e100;
    for i,answer in enumerate(answers):
        vec = glove.getVec(answer);

        # Two word answer, adding the vector
        if(" " in answer): vec = getSumVec(answer.split(" "), glove);

        # Glove does not have the answer in its vocabulary
        if(vec == None):
            if(v): error("Glove does not have the answer \"" + answer + "\" in its vocabulary", False);
            return None;

        if( distfunc(vec, targetvec) < mindist and distfunc(vec, targetvec) < threshold ):
            ind, mindist = i, distfunc(vec, targetvec);

    return answers[ind];

#returns lists of nouns, verbs, and adjectives of sentence
def getPOSVecs(sentence):
    nounVec = []
    verbVec = []
    adjVec = []
    for word in sentence:
        ss = wn.synsets(word)
        if len(ss) < 1 or word in stopwords.words('english'): continue
        pos = str(ss[0].pos())
        if pos == 'n':
            nounVec.append(word)
        elif pos == 'v':
            verbVec.append(word)
        elif pos == 'a':
            adjVec.append(word)
    return nounVec, verbVec, adjVec

#####################################################################################################################
###################################################### MODELS #######################################################
#####################################################################################################################


# Returns answer word based on random chance, given the answers 
def randomModel(question, glove, distfunc=cosine, threshold=1):
    return question.answers[random.randint(0,len(question.answers)) - 1];

# Sentence is an array of words
# Returns answer word by averaging the sentence passed in.
# Returns None if an answer doesn't exist in the glove vocab
# Returns -1 if no answers pass the confidence threshold
def sentenceModel(question, glove, distfunc=cosine, threshold=1):
    targetvec = getAverageVec(question.getSentence(), glove);
    ind, mindist = -1, 10e100;

    return findBestVector(targetvec, question.answers, glove, distfunc, threshold)


# Main method
def main():
    if(v): print "Loading passages...";
    questions = loadQuestions() if train else loadQuestions(qfile="../data/test/questions-test.txt", afile="../data/test/answers-test.txt");

    # Initialize all the external data
    if(v): print "Loading all external data...";
    t = time.time();
    unigrams, bigrams, trigrams = getGrams(path=f);
    glove = Glove(g, delimiter=" ", header=False, quoting=csv.QUOTE_NONE);
    tagger = POSTagger(
            'stanford-postagger/models/english-bidirectional-distsim.tagger', 
            'stanford-postagger/stanford-postagger.jar',
            'utf-8'
        );

    if(v): print "Finished loading all external data in " + str(int(time.time() - start)) + " seconds!"
    if(v): print "Starting program now..."

    models = [
        ("Random", randomModel),
        ("Sentence", sentenceModel)
    ];



    for name, model in models:
        scoring.score_model( [(model(q, glove), q.getCorrectAnswer()) for q in questions], verbose=True, modelname=name)
        



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

    # Report error if called the wrong way
    if("help" in args or train == None):
        error("Example call: python main.py  -v -g ../data/glove_vectors/glove.6B.300d.txt" + 
            "   1) -v: if you want this program to be verbose\n" +
            "   2) -g: path to glove vector file (defaults to '../data/glove_vectors/glove.6B.50d.txt'" + 
            "   3) -train/-test: designates whether you want to evaluate on train or test (required)" + 
            "   4) -f: files to read tect from, default '../data/Holmes_Training_Data/'", True)


    # Loading Modules
    import scipy
    from sklearn import svm
    from nltk.tag.stanford import POSTagger
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
    import numpy as np
    if(v): print "All modules successfully loaded in " + str(int(time.time() - start)) +  " seconds!"

    # Main Method
    main();

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
print glove.getVocab(); # <= returns an array of all the words the glove vectors have
"""






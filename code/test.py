import sys
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

def loadQuestions(directory="../data/train/"):
    files = getRecursiveFiles(directory, lambda x: x[x.rfind("/") + 1] != "." and ".txt" in x and x[-1] != '~' and "norvig" not in x.lower());
    return [Question(text) for filename in files for text in readFile(filename).split("\n\n") ];

def getStrippedAnswerWords(answer):
    answers = filter(lambda x: len(x) > 0 and x not in stopwords.words('english') + ["upon", "toward"], re.split("[ ,]", answer.lower()));
    if(len(answers) > 2):
        print "error:" + answer, answers
    assert(len(answers) <= 2) # checking to make sure correct split
    return answers if len(answers) > 0 else [answer.strip()]; # if answer is a stop word

def testQuestions(questions):
    for question in questions:
        print question
        #test double blank
        if len(getStrippedAnswerWords(question.answers[0])) == 2:
            if len(re.findall( '____(.*?)____', question.text, re.DOTALL)) == 0:
                print "double blank error", question + "\n"
        for answer in question.answers:
            answer_words = getStrippedAnswerWords(answer)
            print answer, answer_words
            assert(len(answer_words) == 1 or len(answer_words) == 2)



testQuestions(loadQuestions(directory="../data/train"))
testQuestions(loadQuestions(directory="../data/dev_set"))
testQuestions(loadQuestions(directory="../data/test"))
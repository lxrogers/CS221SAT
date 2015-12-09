from cayman_utility import *

# All models take in a glove class
# All models take in a question to answer
# All models take in a distance function to use to determine answers (default: cosine)
# All models take in a threshold to use (default: 2)
#   -If the minimum distance between the closest answer we are given and the vector we think
#    the answer should be is greater than the threshold, we don't answer the question. That is,
#    if our answers aren't close (less than the threshold), we return (-1, inf) signaling
#    we will omit the question and the closest distance (which could be inf if we don't understand
#    and answer).
#    
# All models will return tuple(answer they think, closest distance),
#   as defined by the targetvec (if not answer elimination) they generate
#   and findBestVector in utility.py


# Useful globals at bottom of file


#####################################################################################################################
################################################### MODELS ##########################################################
#####################################################################################################################



# Averages adjectives in sentence and uses it as guess to what should be in blank
def adjectiveModel(glove, question, distfunc=cosine, threshold=2, tvec=False):
    nouns, verbs, adjectives = getPOSVecs(question.getSentence());

    # No adjectives in sentence, don't answer
    if(len(adjectives) == 0): return (-1, 2)

    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), adjectives))
    if(tvec): return targetvec

    # Don't understand adjectives in the sentence, don't answer
    if(targetvec == None): return (-1, 2);

    return findBestVector(glove, targetvec, question.answers, distfunc, threshold);



# Averages verbs in sentence and uses it as guess to what should be in blank
def verbModel(glove, question, distfunc=cosine, threshold=2, tvec=False):
    nouns, verbs, adjectives = getPOSVecs(question.getSentence());

    # No verbs found in sentence, don't answer
    if(len(verbs) == 0): return (-1, 2)

    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), verbs))
    if(tvec): return targetvec

    # No verbs we understand in sentence, don't answer
    if(targetvec == None): return (-1, 2)

    # Bug-hunting purposes - ignore
    if(len(targetvec) == 1):
        print question, verbs

    return findBestVector(glove, targetvec, question.answers, distfunc, threshold);



# Averages nouns in sentence and uses it as guess to what should be in blank
def nounModel(glove, question, distfunc=cosine, threshold=2, tvec=False):
    nouns, verbs, adjectives = getPOSVecs(question.getSentence());

    # No nouns found in sentence, don't answer
    if(len(nouns) == 0): return (-1, 2)
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), nouns))
    if(tvec): return targetvec

    # No nouns understood in sentence, don't answer
    if(targetvec == None): return (-1, 2);

    return findBestVector(glove, targetvec, question.answers, distfunc, threshold);



# Averages all words in sentence and uses it as guess to what should be in blank
def sentenceModel(glove, question, distfunc=cosine, threshold=2, tvec=False, unigrams=None):
    targetvec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), question.getSentence()), unigrams);
    if(tvec): return targetvec

    # Understand no words in sentence, don't answer
    if(targetvec == None): return (-1, 2);

    return findBestVector(glove, targetvec, question.answers, distfunc, threshold);



# Averages words in sentence, weighted by unigrams and uses it as guess to what should be in blank
def weightedSentenceModel(glove, question, unigrams=None, distfunc=cosine, threshold=2, tvec=False):
    return sentenceModel(glove, question, distfunc, threshold, tvec, unigrams)


# If double-blank sentence, tries to eliminate answers and use sentence model on new questions
def doubleSentenceModel(glove, question, distfunc=cosine, threshold=2, tvec=False):
    answer_words = getStrippedAnswerWords(question.answers[0])
    if(len(answer_words) == 1):
        #single blank answer
        return sentenceModel(glove, question, distfunc, threshold,tvec=tvec)
    elif(len(answer_words) == 2):
        #double blank answer
        elimination_mode = getDoubleBlankEliminationMode(question) #step 1: use clue words to determine which answers to eliminate (similar or different)
        question2 = getRemainingAnswers(glove, elimination_mode, question) #step 2: eliminate those words
        return sentenceModel(glove, question2, distfunc, threshold, tvec=tvec) #step 3: find best answer out of un-eliminated words

# If double-blank sentence, tries to eliminate answers and use noun model on new questions
def doubleNounModel(glove, question, distfunc=cosine, threshold=2, tvec=False):
    answer_words = getStrippedAnswerWords(question.answers[0])
    if(len(answer_words) == 1):
        #single blank answer
        return nounModel(glove, question, distfunc, threshold,tvec=tvec)
    elif(len(answer_words) == 2):
        #double blank answer
        elimination_mode = getDoubleBlankEliminationMode(question) #step 1: use clue words to determine which answers to eliminate (similar or different)
        question2 = getRemainingAnswers(glove, elimination_mode, question) #step 2: eliminate those words
        return nounModel(glove, question2, distfunc, threshold, tvec=tvec) #step 3: find best answer out of un-eliminated words

# Unusable with distance ML - can't evaluate fit of answer to sentence
def doubleSentenceMaxModel(glove, question, distfunc=cosine, threshold=2, tvec=False):
    answer_words = getStrippedAnswerWords(question.answers[0])
    if(len(answer_words) == 1):
        #single blank answer
        return sentenceModel(glove, question, distfunc, threshold,tvec=tvec)
    elif(len(answer_words) == 2):    
        #double blank answer
        elimination_mode = getDoubleBlankEliminationMode(question) #step 1: use clue words to determine which answers to eliminate (similar or different)
        if elimination_mode == "neutral":
            return sentenceModel(glove, question, distfunc, threshold,tvec=tvec)
        else:
            return getMaxDoubleBlankAnswer(glove, elimination_mode, question)
            
# Unusable with distance ML - can't evaluate fit of answer to sentence
def distanceModel(glove, question, distfunc=cosine, threshold=2, tvec=False):

    bestanswer, mindist = "", float('inf');

    for answer, word in itertools.product(question.answers, filter(lambda x: x not in stopwords.words('english'), question.getSentence())):
        if(answer not in glove or word not in glove): continue;
        dist = distfunc(glove.getVec(answer), glove.getVec(word));
        if(dist < mindist):
            mindist, bestanswer = dist,answer
    return (bestanswer, mindist)


def unigramModel(unigrams, bigrams, question, distfunc=cosine, threshold=2, tvec=False):
    return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: unigrams.score(x[1]));

def bigramModel(unigrams, bigrams, question, distfunc=cosine, threshold=2, tvec=False):
    return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: bigrams.score(x[1]));
# Not used -- too long to load
# def backOffModel(question, distfunc=cosine, threshold=2, tvec=False):
#     if(not rev):
#         return max([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: backoff.score(x[1]));
#     else:
#         return min([(question.answers[i], question.getFilledSentence(i)) for i in xrange(len(question.answers))], key=lambda x: backoff.score(x[1]));


# Not used -- baseline
# Returns answer word based on random chance, given the answers 
# def randomModel(question, distfunc=cosine, threshold=2, tvec=False):
#     return question.answers[random.randint(0,len(question.answers)) - 1];
#     

vsm_models = [
    ("Sentence", sentenceModel),
    ("Distance Model", distanceModel),
    ("Weighted VSM", weightedSentenceModel),
    ("Double Blank Combo VSM", doubleSentenceModel),
    ("Double Blank Max VSM", doubleSentenceMaxModel),
    ("Adjective Model", adjectiveModel),
    ("Noun Model", nounModel),
    ("Verb Model", verbModel),
    ("Double Noun Model", doubleNounModel)
];

targetvec_models = [
    ("Sentence", sentenceModel),
    ("Double Blank Combo VSM", doubleSentenceModel),
    ("Adjective Model", adjectiveModel),
    ("Noun Model", nounModel),
    ("Verb Model", verbModel),
    ("Double Noun Model", doubleNounModel)
];

language_models = [
    ("Unigram", unigramModel),
    ("Bigram", bigramModel)
];

distances = [
    (kldist, "kldist"),
    (jsd, "jsd"),
    (cosine, "cosine"),
    (L2, "L2"),
    (jaccard, "jaccard")
];



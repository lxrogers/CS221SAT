import math, collections

class CustomLanguageModel:
    def __init__(self):
        """Initialize your data structures in the constructor."""
        self.ngramCounts = collections.defaultdict(lambda: 0);
        self.continuationProb = collections.defaultdict(lambda: set());
        self.total = 0;

    def train(self, corpus):
        """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
        """  

        # Generate all possible n-grams
        # for every sentence in the corpus
        for sentence in corpus:

            #for every possible gram-length in the sentence
            for gramlength in xrange(1,len(sentence)):

                #iterate through all possible grams of that gramlength
                for i in xrange(len(sentence) - gramlength):

                    #generate tuple
                    key = ();
                    for index in xrange(gramlength):
                        key += (sentence[i + index],);

                    if(gramlength == 2):
                        self.continuationProb[key[1]].add(key[0]);

                    self.ngramCounts[key] += 1;

        self.total = len(set(map(lambda tup: tup[0], self.ngramCounts)));

    def getBackOff(self, key):
        if(len(key) <= 1): # at the end and have a ngram of length one
            return float(self.ngramCounts[key] +1  + len(self.continuationProb[key]))/(self.total)
        elif(key in self.ngramCounts): # we found the right ngram
            return float(self.ngramCounts[key])/self.ngramCounts[key[:-1]];
        else: # we need to backoff
            return .41*self.getBackOff(key[1:]);

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the 
            sentence using your language model. Use whatever data you computed in train() here.
        """
        s = 0;

        #for every word
        for i in xrange(len(sentence)):
            score =  self.getBackOff(tuple(sentence[:i+1]));
            if(score != 0):
                s += math.log(score);


        return s
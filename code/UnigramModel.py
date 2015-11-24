import math, collections
def zero_fn():
    return 0;
class UnigramModel:

    def __init__(self):
        """Initialize your data structures in the constructor."""
        self.unigramProbs = collections.defaultdict(zero_fn)
        self.unigramCounts = collections.defaultdict(zero_fn);
        self.total = 0;

    def train(self, corpus):
        """ Takes a corpus and trains your language model. 
            Compute any counts or other corpus statistics in this function.
        """  

        for sentence in corpus:
            for word in sentence:
                self.unigramCounts[word] = self.unigramCounts[word] + 1;
                self.total += 1;

        self.unigramCounts["<UNK>"] = 0;

        for key in self.unigramCounts:
            self.unigramProbs[key] = float(self.unigramCounts[key] + 1)/len(self.unigramCounts);

    def getSingleScore(self, word):
        """ Takes a word and returns the log-probability of that word. If not in the dictionary, uses <UNK>.
        """
        if word in self.unigramProbs:
            return math.log(self.unigramProbs[word])
        return math.log(self.unigramProbs["<UNK>"])

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the 
            sentence using your language model. Use whatever data you computed in train() here.
        """
        s = 0;
        for word in sentence:
            if(word in self.unigramProbs):
                s += math.log(self.unigramProbs[word]);
            else:
                s += math.log(self.unigramProbs["<UNK>"]);

        return s;

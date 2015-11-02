import math, collections

class LaplaceUnigramLanguageModel:

    def __init__(self):
        """Initialize your data structures in the constructor."""
        self.unigramProbs = collections.defaultdict(lambda: 0)
        self.unigramCounts = collections.defaultdict(lambda: 0);
        self.total = 0;

    def train(self, corpus):
        """ Takes a corpus and trains your language model. 
            Compute any counts or other corpus statistics in this function.
        """  

        for sentence in corpus:
            for word in sentence:
                unigramCounts[word] = unigramCounts[word] + 1;
                self.total += 1;


        unigramCounts["<UNK>"] = 0;

        for key in unigramCounts:
            self.unigramProbs[key] = float(unigramCounts[key] + 1)/len(unigramCounts);


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
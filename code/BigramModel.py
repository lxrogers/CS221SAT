import collections, math
def zero_fn():
    return 0;
class BigramModel:

    def __init__(self):
        """Initialize your data structures in the constructor."""
        self.bigramCounts = collections.defaultdict(zero_fn);
        self.unigramCounts = collections.defaultdict(zero_fn);

    def train(self, corpus):
        """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
        """  


        for sentence in corpus:
            for i in xrange(len(sentence) - 1):  
                word1 = sentence[i];
                word2 = sentence[i + 1];
                self.bigramCounts[(word1, word2)] += 1;
                self.unigramCounts[word1] += 1;

            self.unigramCounts[sentence[len(sentence) - 1]] += 1;
        

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
        """
        s = 0;
        for i in xrange(len(sentence) - 1):
            word1 = sentence[i];
            word2 = sentence[i+1];
            pair = (word1, word2);
            if(self.bigramCounts[pair] > 0): #have seen before
                s += math.log( float(self.bigramCounts[pair] + 1)/(self.unigramCounts[word1] + len(self.unigramCounts)));
            else:
                if(word1 in self.unigramCounts):
                    s += math.log( 1.0/(len(self.unigramCounts) + self.unigramCounts[word1]));
                else:
                    s += math.log( 1.0/len(self.unigramCounts));
        return s
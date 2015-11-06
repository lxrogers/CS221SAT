#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: Question.py

import re

class Question:

    def __init__(self, string):
        """Question initialization."""
        arr = filter(lambda x: len(x) > 0, string.split("\n"));
        self.text = arr[0].strip();
        self.answers = map(lambda x: x.strip(), arr[1:-1]);
        self.correctAnswer = int(arr[-1]);

    def __str__(self):
        return "Question:\n\t" + self.text + "\n\t" + str(self.answers) + \
        "\n\tCorrect Answer: " + str(self.correctAnswer) + "; " + self.answers[self.correctAnswer];

    def __repr__(self):
        return str(self);

    def getSentence(self):
    	return map(lambda x: x.strip().lower(), filter(lambda x: len(x) > 0,  re.split("[^A-Za-z0-9_\']", self.text)));

    def getFilledSentence(self, index):
        if(len(self.answers[index].split(",")) <= 1):
            return self.text.replace("____", self.answers[index]);
        else:
            try:
                answers = map(lambda x: x.strip(), self.answers[index].split(","));
                temp = self.text.replace("____", answers[0],1)
                return temp.replace("____", answers[1], 1);
            except:
                print answers;

    def getCorrectWord(self):
    	return self.answers[self.correctAnswer];
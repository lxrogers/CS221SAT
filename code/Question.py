#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: Question.py

import re

# The class in which we store all the question information
class Question:

    def __init__(self, string):
        """Question initialization."""

        # The files with which we read the questions must be well-formed
        arr = filter(lambda x: len(x) > 0, string.split("\n"));

        # Get the Question (sentence) text
        self.text = arr[0].strip();

        # Get all the answers
        self.answers = map(lambda x: x.strip(), arr[1:-1]);

        # Validate that we're reading the questions correctly
        for answer in self.answers:
            assert(len(re.split("[ ,]", answer)) <= 4 and len(answer) > 0);

        # Get which answer is correct
        self.correctAnswer = int(arr[-1]);

    # Convert !uestion to string for debugging purposes
    def __str__(self):
        return "Question:\n\t" + self.text + "\n\t" + str(self.answers) + \
        "\n\tCorrect Answer: " + str(self.correctAnswer) + "; " + self.answers[self.correctAnswer];

    # Printing Question for debugging purposes
    def __repr__(self):
        return str(self);

    # Parses the Question text and returns a tokenized version of the Question sentence
    def getSentence(self):
    	return map(lambda x: x.strip().lower(), filter(lambda x: len(x) > 0,  re.split("[^A-Za-z0-9_\']", self.text)));

    # Fills the sentence with the answer choice chosen
    def getFilledSentence(self, index):
        if(len(self.answers[index].split(",")) <= 1):
            return self.text.replace("____", self.answers[index]);
        else:
            try:
                answers = map(lambda x: x.strip(), self.answers[index].split(","));
                temp = self.text.replace("____", answers[0],1)
                return temp.replace("____", answers[1], 1);
            except:
                # For debugging if there is a malformed question
                print answers;

    # Helper method that returns the correct word -- helps verbosity in other files
    def getCorrectWord(self):
    	return self.answers[self.correctAnswer];
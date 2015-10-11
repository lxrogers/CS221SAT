#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: Question.py

import re

class Question:

    def __init__(self, prompt, number, answers, correct):
        """Question initialization."""
        self.text = prompt;
        self.number = number
        self.answers = answers;
        self.correctAnswer = correct;

    def __str__(self):
        return "Question #" + self.number + ":\n\t" + self.text + "\n\t" + str(self.answers) + \
        "\n\tCorrect Answer: " + str(self.correctAnswer) + "; " + self.answers[self.correctAnswer]

    def getSentence(self):
    	return map(lambda x: x.strip().lower(), filter(lambda x: len(x) > 0,  re.split("[^A-Za-z0-9]", self.text)));

    def getCorrectAnswer(self):
    	return self.answers[self.correctAnswer];
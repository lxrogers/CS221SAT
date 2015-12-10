#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: score.py

from cayman_utility import *
from cayman_models import *
from Glove import *
from Question import *
import scoring

# Script to analyze our elementary models on the dev set and test set
def score(question_dir = "../data/cayman_all_training.txt", glove_file="../data/glove_vectors/glove.6B.300d.txt", ngram_path="../data/Holmes_Training_Data/norvig.txt", dev=True):
    print "Training N-Grams" # Load/Generate N-grams
    unigrams, bigrams, cgrams = getGrams(path=ngram_path)

    print "Loading Questions" # Load questions
    questions = loadQuestions(question_dir)

    # Holds questions to be evaluated
    eval_qs = None

    if dev:
        # Split into train/dev
        split = len(questions) - len(questions)/10;
        inform("Splitting Data: " + str(split) + " questions in training and " + str(len(questions) - split) + " in dev...");
        train_questions, eval_qs = questions[:split], questions[split:];
    else:
        eval_qs = questions

    print "Loading Glove" # Loads Glove vectors
    glove = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False)

    # For every VSM model
    for name, model in vsm_models:

        # We get the model's score
        print "Scoring ", name
        answer_guess_pairs = []
        for question in eval_qs:
            guess = None

            # Weighted VSM has an extra parameter
            if name == "Weighted VSM":
                guess = model(glove, question, unigrams)[0]
            else:
                guess = model(glove, question, threshold=.95)[0]

            # Get the correct answer
            answer = question.getCorrectWord()

            # Add to tuple GOLD and guessed answers
            answer_guess_pairs.append((guess, answer))
		
        print "\n\n"
        scoring.score_model(answer_guess_pairs, verbose=True, modelname=name)

    # Now score Language models
    # For every Language model
    for name, model in language_models:

        # Do the same thing as before
        print "Scoring ", name
        answer_guess_pairs = []

        # For every question
        for question in eval_qs:

            # Generate guess from model
            guess = model(unigrams, bigrams, question)[0]

            # Find GOLD answer (correct answer)
            answer = question.getCorrectWord()

            # Add tuple for scoring
            answer_guess_pairs.append((guess, answer))
		
        print "\n\n"
        scoring.score_model(answer_guess_pairs, verbose=True, modelname=name)

# Call the method
score()

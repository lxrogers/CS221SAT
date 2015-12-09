from cayman_utility import *
from cayman_models import *
from Glove import *
from Question import *
import scoring

# Script to analyze our elementary models on the dev set and test set


def score(question_dir = "../data/cayman_all_training.txt", glove_file="../data/glove_vectors/glove.6B.300d.txt", ngram_path="../data/Holmes_Training_Data/norvig.txt", dev=True):
    print "Training N-Grams"
    unigrams, bigrams, cgrams = getGrams(path=ngram_path)

    print "Loading Questions"
    questions = loadQuestions(question_dir)
    eval_qs = None
    if dev:
        # Split into train/dev
        split = len(questions) - len(questions)/10;
        inform("Splitting Data: " + str(split) + " questions in training and " + str(len(questions) - split) + " in dev...");
        train_questions, eval_qs = questions[:split], questions[split:];
    else:
        eval_qs = questions

    print "Loading Glove"
    glove = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False)

    for name, model in vsm_models:
        print "Scoring ", name
        answer_guess_pairs = []
        for question in eval_qs:
            guess = None
            if name == "Weighted VSM":
                guess = model(glove, question, unigrams)[0]
            else:
                guess = model(glove, question)[0]
            answer = question.getCorrectWord()
            answer_guess_pairs.append((guess, answer))
		
        print "\n\n"
        scoring.score_model(answer_guess_pairs, verbose=True, modelname=name)
        #create answer, geuss pair for each model

    for name, model in language_models:
        print "Scoring ", name
        answer_guess_pairs = []
        for question in eval_qs:
            guess = model(unigrams, bigrams, question)[0]
            answer = question.getCorrectWord()
            answer_guess_pairs.append((guess, answer))
		
        print "\n\n"
        scoring.score_model(answer_guess_pairs, verbose=True, modelname=name)
        #create answer, geuss pair for each model

score()

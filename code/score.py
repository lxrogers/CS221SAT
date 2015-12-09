from cayman_utility import *
import cayman_models
from Glove import *
from Question import *
import scoring

def score(question_dir = "../data/dev_set/devset_65questions.txt", glove_file="../data/glove_vectors/glove.6B.300d.txt"):

	print "Loading Questions"
	questions = loadQuestions(question_dir)
	
	print "Total Questions: " , len(questions)
	print "Loading Glove THREE HUNDRED"
	glove = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False)

	models = [
		("Sentence", cayman_models.sentenceModel),
        ("Distance Model", cayman_models.distanceModel),
        ("Weighted VSM", cayman_models.weightedSentenceModel),
        ("Double Blank Combo VSM", cayman_models.doubleSentenceModel),
        ("Double Blank Max VSM", cayman_models.doubleSentenceMaxModel),
        ("Adjective Model", cayman_models.adjectiveModel),
        ("Noun Model", cayman_models.nounModel),
        ("Verb Model", cayman_models.verbModel)
	]

	total = 100#len(questions) * len(models)
	for model in models:
		print "Scoring ", model[0]
		count = 0
		answer_guess_pairs = []
		for question in questions:
			guess = model[1](glove, question)[0]
			answer = question.getCorrectWord()
			answer_guess_pairs.append((guess, answer))
			count += 1

			if (len(answer_guess_pairs) > 100):
				break;
			if count % (total / 100) == 0:
				print round(count * 1.0 / total, 2), "% ", 

		print "\n\n"
		scoring.score_model(answer_guess_pairs, True, model[0])
		#create answer, geuss pair for each model

score("../data/cayman_all_training.txt")
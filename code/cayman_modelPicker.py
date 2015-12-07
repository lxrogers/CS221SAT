import cayman_models as models
from cayman_utility import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.qda import QDA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import scoring

algorithms = [
		  (RandomForestClassifier(max_depth=5, n_jobs=-1, n_estimators=10, max_features=10), "Random Forest"),
		  (GaussianNB(), "Gaussian Naive Bayes"),
		  (LogisticRegression(), "Logistic Regression"),
		  (LinearSVC(), "Support Vector Machine"),
		  (DecisionTreeClassifier(max_depth=10), "Decision Tree"),
		  #(QDA(), "QDA"),
		  (GradientBoostingClassifier(), "BOOSTING!!!"),
		  (Pipeline(steps=[('rbm', BernoulliRBM()), ('logistic', LogisticRegression())]), "Bernoulli Neural Network Combo Logit")
		]






# Create dataset of (X, y) where X is questions, y is labels.
#	 In this case, y is the label of which algorithm to use to answer the question based on
#	 sentence features.
def generateDataset(datafile):
	generatedFile = "../data/cayman_sentence_data/sentenceDataset.pickle"

	# No need to generate twice. 
	if(isfile(generatedFile)):
		return loadPickle(generatedFile);

	# If we haven't generated the dataset yet
	questions = loadQuestions(datafile);

	# For now, we will use regular parameters to generate dataset.
	# The y will be the name of the model who gets it closest.
	y = [];
	for question in questions:
		# For every question, we're evaluating all the models
		# and we're choosing the model who most confidently gets it (smallest distance)
		bestModel, smallestDistance = "No model", 2

		for name, model in models.vsm_models:
			# What the model guesses and the distance it thinks the perfect answer is from
			# the given answers in the SAT qustion
			guess, dist = model(glove, question);

			# Model failed at guessing question
			if(guess == -1 or guess == None): continue;

			# The model that gets the question right, and gets it best wins
			if(question.getCorrectWord() == guess and dist < smallestDistance):
				bestModel, smallestDistance = name, dist

		# At this point, we've tried all the models and we have the best model that performed
		# on that question with the distance it got
		y.append(bestModel);

	# Save the X,y pair so we don't have to generate the dataset again
	savePickle( (questions, y), "../data/cayman_sentence_data/sentenceDataset.pickle")
	return (questions, y); 
		






# After creating the dataset, we featurize X for the ML algorithm. Returns (phi(X), y)
#	 for training. We use many basic sentence features to try to gain the most information
#	 about which model to use (i.e. the label, y).
def featurize(q):

	# Constants defining features
	SUPPORT_WORDS = ["moreover", "besides", "additionally", "furthermore", "in fact", "and", "therefore"]
	CONTRAST_WORDS = ["although", "however", "rather than", "nevertheless", "whereas", "on the other hand", "but"]
	NUM_FEATURES = 15
	NOUNS_INDEX = 0
	ADJECTIVES_INDEX = 1
	VERBS_INDEX = 2
	NEGATION_INDEX = 3 # Currently do not support negation feature
	SUPPORT_INDEX = 4
	CONTRAST_INDEX = 5
	SEMICOLON_INDEX = 6
	TOTAL_WORDS_INDEX = 7
	CAPITAL_WORDS_INDEX = 8
	BLANK_POSITION_INDEX = 9
	COMMAS_INDEX = 10
	COLON_INDEX = 11
	BLANK_PERCENT_INDEX = 12 # position of blank as percentage of total words
	IS_DOUBLE = 13
	EXCLAMATION_INDEX = 14

	# The sentence of the question in an array
	sentence = q.getSentence()

	# Features to be returned
	features = [0]*NUM_FEATURES

	# Part of Speech vectors defined in cayman_utility.py
	nouns, verbs, adjectives = getPOSVecs(sentence)

	features[NOUNS_INDEX] = len(nouns);
	features[VERBS_INDEX] = len(verbs);
	features[ADJECTIVES_INDEX] = len(adjectives);
	features[TOTAL_WORDS_INDEX] = len(sentence);
	features[SUPPORT_INDEX] = len(filter(lambda x: x in SUPPORT_WORDS, sentence));
	features[CONTRAST_INDEX] = len(filter(lambda x: x in CONTRAST_WORDS, sentence));
	features[COMMAS_INDEX] = 1 if "," in q.text else 0; # TODO: we could do count, not just indicator?
	features[SEMICOLON_INDEX] = 1 if ";" in q.text else 0;
	features[COLON_INDEX] = 1 if ":" in q.text else 0;
	features[EXCLAMATION_INDEX] = 1 if "!" in q.text else 0;
	features[CAPITAL_WORDS_INDEX] = len(filter(lambda x: len(x) > 0 and x[0].isalpha() and x[0].isupper(), sentence)) - 1;

	# Get Double Blank and First Blank Position
	try:
		features[BLANK_POSITION_INDEX] = sentence.index('____')
		features[BLANK_PERCENT_INDEX] = features[BLANK_POSITION_INDEX] * 1.0/ features[TOTAL_WORDS_INDEX]
	except:
		features[BLANK_POSITION_INDEX] = 0
	try:
		if sentence.count('____') > 1:
			features[IS_DOUBLE] = 1
	except:
		features[IS_DOUBLE] = 0

	return features








# Train the ML algorithms on (phi(x), y)
def train(phi, y):
	for algorithm, name in algorithms:
		print "Training", name + "...";
		algorithm.fit(phi, y);








# Evaluate the ML algorithms on the trained algorithms
def evaluateML(phi, y):
	for algorithm, name in algorithms:
		print '{0}{1}: {2}'.format(name, ' '*(36 - len(name)), str(algorithm.score(phi, y)));









# Tests the ML algorithm's model choice on questions
def evaluateScore(questions, features, labels):
	vsm_models = dict(models.vsm_models);

	# For every ML Algorithm we trained...
	for algorithm, name in algorithms:

		# Go through and using the model it thinks will guess right, guess the question
		guesses = [];
		for question, phi, label in zip(questions, features, labels):

			# The model the algorithm thinks we should use
			prediction =  algorithm.predict(phi)[0];

			# If we predict we can't get this question right
			if(prediction == "No model"): guesses.append((-1, 0));
			else:
				# Get the model we're going to use
				model = vsm_models[algorithm.predict(phi)[0]];

				# Using the model to answer the question
				guesses.append((model(glove, question)[0], question.getCorrectWord()));

		# How did this ML algorithm do?
		scoring.score_model(guesses, verbose=True, modelname=name);










# Main method
def main():

	# Create or Load the dataset in -- X is array of questions, y is labels (name of model to use)
	inform("Generating/Loading dataset...");
	X, y = generateDataset("../data/cayman_all_training.txt");

	# Convert array of Questions (X) into sentence features phi(X)
	inform("Featurizing all " + str(len(X)) + " questions...");
	phi = map(lambda x: featurize(x), X);

	# Split into train/dev
	split = len(X) - len(X)/10;
	inform("Splitting Data: " + str(split) + " questions in training and " + str(len(X) - split) + " in dev...");
	train_questions, dev_questions = X[:split], X[split:];
	train_features, dev_features = phi[:split], phi[split:];
	train_labels, dev_labels = y[:split], y[split:];

	inform("Training Machine Learning algorithms...");
	train(train_features, train_labels);

	inform("Training Error");
	evaluateML(train_features, train_labels);

	inform("Dev Error");
	evaluateML(dev_features, dev_labels);

	inform("Evaluating Score of ML algorithms choosing models on Training Data");
	evaluateScore(train_questions, train_features, train_labels);

	inform("Evaluating Score of ML algorithms choosing models on Dev");
	evaluateScore(dev_questions, dev_features, dev_labels);

# Boilerplate code
if __name__ == "__main__":

	# Using standard glove for now to just get this working...
	inform("Finished importing! Loading Glove...");
	glove = Glove("../data/glove_vectors/glove.6B.50d.txt", delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False);

	main();
import warnings
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


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

algorithms = [
		  (RandomForestClassifier(max_depth=5, n_jobs=-1, n_estimators=10, max_features=10), "Random Forest"),
		  (GaussianNB(), "Gaussian Naive Bayes"),
		  (LogisticRegression(), "Logistic Regression"),
		  (LinearSVC(), "Support Vector Machine"),
		  (DecisionTreeClassifier(max_depth=10), "Decision Tree"),
		  (QDA(), "QDA"),
		  (GradientBoostingClassifier(), "BOOSTING!!!"),
		  (Pipeline(steps=[('rbm', BernoulliRBM()), ('logistic', LogisticRegression())]), "Bernoulli Neural Network Combo Logit")
		]






# Create dataset of (X, y) where X is a list of (answer, question), y is labels.
#	 In this case, y is the label of whether the answer is correct for the question
def generateDataset(datafile):
	generatedFile = "../data/cayman_distance_data/distanceDataset.pickle"

	# No need to generate twice. 
	if(isfile(generatedFile)):
		return loadPickle(generatedFile);

	# If we haven't generated the dataset yet
	questions = loadQuestions(datafile);

	# For now, we will use regular parameters to generate dataset.
	# The y will be the name of the model who gets it closest.
	X, y = [], [];
	for question in questions:
		for i, answer in enumerate(question.answers):
			X.append( (answer, question) );
			y.append(1 if i == question.correctAnswer else 0);

	# Save the X,y pair so we don't have to generate the dataset again
	savePickle( (X, y), generatedFile)
	return (X, y); 
		






# After creating the dataset, we featurize X for the ML algorithm. Returns (phi(X), y)
#	 for training. We use many basic sentence features to try to gain the most information
#	 about which model to use (i.e. the label, y).
def featurize(X):
	featureFile = "../data/cayman_distance_data/featureDataset.pickle";
	gloveFile = "../data/glove_vectors/glove.6B.50d.txt";

	if(isfile(featureFile)):
		return loadPickle(featureFile);

	weightings = ["pmi", "ppmi", "tfidf", "None"];
	lsa = []; #[250,100,50,25]

	features = [[1]]*len(X); # Starting out with bias features

	# For every type of weighting for the glove vectors
	for weighting in weightings:

		# Load the glove vector (once)
		print "Featurizing with glove with weight " + weighting + "...";
		glove =  Glove(gloveFile, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, weighting=weighting, v=False);

		# For every type of model
		for name, model in models.targetvec_models:

			# For every type of distance metric
			for distance, dist_name in models.distances:

				# For every answer/question combo
				for i, (answer, question) in enumerate(X):
					answervec = glove.getVec(answer);
					targetvec = model(glove, question, tvec=True)
					if(answervec == None or targetvec == None or len(answervec) != targetvec):
						features[i] += [float('inf')];
					else:
						d = distance(targetvec, answervec);
						features[i] += [d if not d.isnan() else float('inf')]



	savePickle(features, featureFile);
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
	phi = featurize(X);

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

	# inform("Evaluating Score of ML algorithms choosing models on Training Data");
	# evaluateScore(train_questions, train_features, train_labels);

	# inform("Evaluating Score of ML algorithms choosing models on Dev");
	# evaluateScore(dev_questions, dev_features, dev_labels);

# Boilerplate code
if __name__ == "__main__":
	main();
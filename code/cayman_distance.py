# Imports
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
import math

# Filter out all the warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# All the ML algorithms we're going to try
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
    gloveFile = "../data/glove_vectors/glove.6B.300d.txt";

    if(isfile(featureFile)):
        return loadPickle(featureFile);

	#weightings = ["pmi", "ppmi", "tfidf", "None"];
    weightings = ["tfidf", "None"];
    lsa = []; #[250,100,50,25]

    features = [[1]]*len(X); # Starting out with bias features

    # For every type of weighting for the glove vectors
    for weighting in weightings:

        # Load the glove vector (once)
        print "Featurizing with glove with weight " + weighting + "...";
        glove = None
        if weighting == "None":
            glove = Glove(gloveFile, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False);
        else:
            glove =  Glove(gloveFile, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, weighting=weighting, v=False);

		# For every type of model
        for name, model in models.targetvec_models:
            
            # Weighted VSM is the same as Sentence VSM because we don't use unigram counts
            if (name == "Weighted VSM"): continue;

			# For every type of distance metric
            for distance, dist_name in models.distances:
                if (dist_name == "kldist" or dist_name == "jsd" or dist_name == "jaccard"): continue;
				# For every answer/question combo
                for i, (answer, question) in enumerate(X):

                    # Get the vector that the model predicts should go in the blank
                    targetvec = model(glove, question, tvec=True)

                    # If the answer is not in the Glove dictionary, we say it's max distance
                    if (targetvec == None or isinstance(targetvec, tuple)):
                        features[i].append(2)

                    # If the distance is valid, we add it as a feature
                    else:
                        answer_dist = distanceSingleWords(glove, targetvec, answer, distance)
                        features[i].append(answer_dist if not math.isnan(answer_dist) else 2)

    # Save our feature set
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
def evaluateScore(pairs, features, labels):
    vsm_models = dict(models.vsm_models);

	# For every ML Algorithm we trained...
    for algorithm, name in algorithms:

        # For every question (it has 5 answers)
        num_eval = len(pairs)/5;
        guesses = [];
        for i in range(num_eval):
            correct = []

            # Go through each answer/question pair
            for (answer, question), phi, label in zip(pairs[i*5:i*5+5], features[i*5:i*5+5], labels[i*5:i*5+5]):

                # Get prediction
                prediction =  algorithm.predict(phi)[0];

                # If we predict the answer is right
                if prediction == 1:
                    correct.append(answer)
            
            # If we think no answer is right, we omit to answer the question
            if len(correct) == 0:
                guesses.append((-1, 0));

            # Otherwise, we choose the first model that gets the answer right
            else:
                guesses.append((correct[0], question.getCorrectWord()))

		# How did this ML algorithm do? (Evaluation)
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
    
    print len(dev_labels)/5
    inform("Training Machine Learning algorithms...");
    train(train_features, train_labels);

    inform("Training Error");
    evaluateML(train_features, train_labels);

    inform("Dev Error");
    evaluateML(dev_features, dev_labels);

	# inform("Evaluating Score of ML algorithms choosing models on Training Data");
    evaluateScore(train_questions, train_features, train_labels);

	# inform("Evaluating Score of ML algorithms choosing models on Dev");
    evaluateScore(dev_questions, dev_features, dev_labels);

# Boilerplate code
if __name__ == "__main__":
	main();

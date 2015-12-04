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
import get_distance_ml_training
import numpy
import scoring

models = [(RandomForestClassifier(max_depth=5, n_jobs=-1, n_estimators=10, max_features=10), "Random Forest"),
          (GaussianNB(), "Gaussian Naive Bayes"),
          (LogisticRegression(), "Logistic Regression"),
          (LinearSVC(), "Support Vector Machine"),
          (DecisionTreeClassifier(max_depth=10), "Decision Tree"),
          (QDA(), "QDA"),
          (GradientBoostingClassifier(), "BOOSTING!!!"),
          (Pipeline(steps=[('rbm', BernoulliRBM()), ('logistic', LogisticRegression())]), "Bernoulli Neural Network Combo Logit")
         ]

def trainDataDevTest(dev=True):
    train_data = None
    eval_data = None
    if dev:
        train_data, eval_data = get_distance_ml_training.getEvaluatingTrainingData();
    else:
        train_data, eval_data = get_distance_ml_training.getTestingTrainingData();
    train = train_data[0]
    train_labels = train_data[1]
    evals = eval_data[0]
    eval_labels = eval_data[1]
    
    num_dev = len(evals)/5

    print "Training the models...";

    for model, name in models:
	    model.fit(train, train_labels);

    print "Get Training Error..."
    for model, name in models:
        print "\nML Algorithm Training: ", name;
        print "Scored: ", model.score(train, train_labels);

    print "Evaluating Models On Dev..."
    for model, name in models:
        num_right = 0
        num_not_answer = 0
        num_wrong = 0;
        for i in range(num_dev):
            vals = model.predict(evals[i*5:i*5+5])
            if 1 in vals:
                pred_index = numpy.where(vals==1)[0][0]
                answer_index = eval_labels[i*5:i*5+5].index(1)
                if pred_index == answer_index:
                    num_right += 1
                else:
                	num_wrong += 1
            else:
                num_not_answer += 1
        print "\nML Algorithm Dev: ", name;
        print "Answered Correctly: %d Did Not Answer: %d" %(num_right, num_not_answer)
	    print "Percent Right: ", model.score(evals, eval_labels);
	    print "SAT Score: ", scoring.score_model([(1,1)]*num_right + [(None,1)]*num_not_answer + [(0,1)]*num_wrong);

trainDataDevTest()

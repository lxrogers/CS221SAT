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

models = [(RandomForestClassifier(max_depth=5, n_jobs=-1, n_estimators=10, max_features=10), "Random Forest"),
          (GaussianNB(), "Gaussian Naive Bayes"),
          (LogisticRegression(), "Logistic Regression"),
          (LinearSVC(), "Support Vector Machine"),
          (DecisionTreeClassifier(max_depth=10), "Decision Tree"),
          (QDA(), "QDA"),
          (GradientBoostingClassifier(), "BOOSTING!!!"),
          (Pipeline(steps=[('rbm', BernoulliRBM()), ('logistic', LogisticRegression())]), "Bernoulli Neural Network Combo Logit")
         ]

def trainDataDev():

    train_data, dev_data = get_distance_ml_training.getEvaluatingTrainingData();
    train = train_data[0]
    train_labels = train_data[1]
    dev = dev_data[0]
    dev_labels = dev_data[1]
    
    num_dev = len(dev)/5

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
        for i in range(num_dev):
            vals = model.predict(dev[i*5:i*5+5])
            if 1 in vals:
                pred_index = numpy.where(vals==1)[0][0]
                answer_index = dev_labels[i*5:i*5+5].index(1)
                if pred_index == answer_index:
                    num_right += 1
            else:
                num_not_answer += 1
        print "\nML Algorithm Dev: ", name;
        print "Answered Correctly: %d Did Not Answered: %d" %(num_right, num_not_answer)
         
    for model, name in models:
	    print "\nML Algorithm Dev: ", name;
	    print "Scored: ", model.score(dev, dev_labels);

trainDataDev()

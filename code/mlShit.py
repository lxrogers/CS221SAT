from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.qda import QDA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import BernoulliRBM
import get_distance_ml_training

models = [(RandomForestClassifier(max_depth=5, n_jobs=-1, n_estimators=10, max_features=10), "Random Forest"),
          (GaussianNB(), "Gaussian Naive Bayes"),
          (LogisticRegression(), "Logistic Regression"),
          (LinearSVC(), "Support Vector Machine"),
          (DecisionTreeClassifier(max_depth=10), "Decision Tree"),
          (QDA(), "QDA"),
          (GradientBoostingClassifier, "BOOSTING!!!"),
          (BernoulliRBM, "Bernoulli Neural Network")
         ]

train, labels = get_distance_ml_training.getTrainingData();

split = int(len(train) - len(train)/10);
train, dev = train[:split], train[split:];
labels, dev_labels = labels[:split], train[split:];


print "Training the models...";

for model, name in models:
	model.fit(train, labels);

print "Evaluating Models..."

for model, name in models:
	print "\nML Algorithm: ", name;
	print "Scored: ", model.score(dev, dev_labels);

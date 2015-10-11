# CS221SAT

AI Final Project to answer sentence completion questions in the Critical Reading section of the SAT

By Cayman Simpson, Orion Despo & Lawrence Rogers

Dependencies:
NLTK - 
	Documentation: http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford
	
	sudo easy_install pip / sudo pip install -U numpy / sudo pip install -U nltk

scikit-learn - 
	Documentation: http://scikit-learn.org/stable/
	
	pip install -U numpy scipy scikit-learn 


COMMAND LINE REFERENCE
===========================================================================

An example call to this program would be:

Example call: python main.py  -v -g ../data/glove_vectors/glove.6B.300d.txt -f ../data/Holmes_Training_Data/norvig.txt -train
   1) -v: if you want this program to be verbose
   2) -g: filename for glove vectors, default "../data/glove_vectors/glove.6B.50d.txt"
   3) -train/-test: designates whether you want to evaluate on train or test (required)
   4) -f: files to read text from, default "../data/Holmes_Training_Data/", do '../data/Holmes_Training_Data/norvig.txt' for shorter load times

Observe the bottom of main.py file for details on how the terminal calling interacts with the test-taking model itself.


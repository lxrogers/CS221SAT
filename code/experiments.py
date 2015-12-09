from Question import *
from cayman_utility import *
from Glove import *

def convertStringToVec(string):
	return map(lambda x: x.strip().lower(), filter(lambda x: len(x) > 0,  re.split("[^A-Za-z0-9_\']", string)));

#let's find out if there are different parts of the sentence that match the CORRECT answer very well
def experiment1(glove_file="../data/glove_vectors/glove.6B.100d.txt", question_dir="../data/all_sat/seven_sat_raw.txt"):
	def getBeforeBlankText(sentence):
		return sentence[:sentence.find("____")]

	def getAfterBlankText(sentence):
		return sentence[sentence.find("____") + len("____"):]

	print "Loading Questions"
	questions = loadQuestions(question_dir)

	print "num questions: " , len(questions)

	print "Loading Glove None"
	glove = Glove(glove_file, delimiter=" ", header=False, quoting=csv.QUOTE_NONE, v=False)

	

	print "Experimenting on 100 percent of questions" 
	for i in range(int(math.floor(len(questions) * 1))):#change 1 to decimal to reduce amount of questions
		question = questions[i]
		#only want single blanks for now
		if len(re.findall ( '____(.*?)____', question.text, re.DOTALL)) != 0:
			continue

		answer_words = getStrippedAnswerWords(question.getCorrectWord())
		answer_vec = glove.getVec(answer_words[0])
		
		total_vec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), question.getSentence()))
		before_vec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), getBeforeBlankText(question.text)))
		after_vec = glove.getAverageVec(filter(lambda x: x not in stopwords.words('english'), getAfterBlankText(question.text)))
		

		total_distance = cosine(answer_vec, total_vec)
		before_distance = cosine(answer_vec, before_vec) if len(before_vec) > 2 else 2
		after_distance = cosine(answer_vec, after_vec) if len(after_vec) > 2 else 2
		if total_distance < before_distance and total_distance < after_distance:
			continue #comment this out to print for every question
		print question.text, answer_words[0]
		print "total distance:", total_distance
		print "before distance: " , before_distance
		print "after distance: " , after_distance
		print "\n\n"

experiment1()
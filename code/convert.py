import re
# Reads a file and returns the text contents
def readFile(filename):
    with open(filename) as f: return f.read();

def convertDataSet(questions_file, answers_file):
	#read file
	questions = readFile(questions_file).split('\n\n\n')
	answers = readFile(answers_file).split('\n')
	newDataSet = ""
	for index in range(0, len(answers) - 1):
		newDataSet += convertQuestion(questions[index], answers[index])
	#print newDataSet
	f = open('../data/train/converted-data.txt', 'w')
	f.write(newDataSet)

def cleanAnswer(answer):
	return answer[answer.find(')') + 2:]

def getAnswerNumber(correctAnswer):
	answerLetter = correctAnswer[correctAnswer.find('[') + 1:correctAnswer.find(']')]
	if answerLetter is 'a': return 0
	if answerLetter is 'b': return 1
	if answerLetter is 'c': return 2
	if answerLetter is 'd': return 3
	if answerLetter is 'e': return 4	


def convertQuestion(questionChunk, correctAnswer):
	arr = filter(lambda x: len(x) > 0, questionChunk.split("\n"))
	question = ""
	sentence = arr[0].strip()
	sentence = sentence[sentence.find(')') + 2:]
	answers = map(lambda x: x.strip(), arr[1:])

	question += sentence + "\n"
	for answer in answers:
		question += cleanAnswer(answer) + "\n"

	question += str(getAnswerNumber(correctAnswer)) + "\n\n"
	return question

convertDataSet('../data/train/holmes-questions.txt', '../data/train/holmes-answers.txt')
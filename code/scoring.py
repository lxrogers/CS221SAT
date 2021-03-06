#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Orry Despo (odespo@stanford.edu), Lawrence Rogers (lxrogers@stanford.edu)
# CS221, Created: 10 October 2015
# file: scoring.py

# Converstion table of raw score to SAT score by straight conversion of
# raw score to standard deviation, as defined by the College Board guidelines
SCORE_CONVERSION_TABLE = dict([(67, 800), (31, 500), (66, 800), (30, 500), 
                               (65, 800), (29, 490), (64, 790), (28, 480), 
                               (63, 770), (27, 480), (62, 760), (26, 470),
                               (61, 740), (25, 460), (60, 730), (24, 460), 
                               (59, 720), (23, 450), (58, 700), (22, 440), 
                               (57, 690), (21, 440), (56, 680), (20, 430),
                               (55, 670), (19, 420), (54, 670), (18, 410), 
                               (53, 660), (17, 410), (52, 650), (16, 400), 
                               (51, 640), (15, 390), (50, 630), (14, 380),
                               (49, 620), (13, 380), (48, 620), (12, 370), 
                               (47, 610), (11, 360), (46, 600), (10, 350), 
                               (45, 600), (9, 340), (44, 590), (8, 330),
                               (43, 580), (7, 320), (42, 570), (6, 310), 
                               (41, 570), (5, 300), (40, 560), (4, 290), 
                               (39, 550), (3, 270), (38, 550), (2, 260),
                               (37, 540), (1, 240), (36, 530), (0, 220), 
                               (35, 530), (-1, 210), (34, 520), (-2, 200),
                               (33, 520), (32, 510)])

# Scores a model based on its guesses
def score_model(guess_answer_pairs, verbose=False, modelname="Name Not Given"):

    # Print the model name
    if(verbose): print '\033[95mModel: ' + modelname + "\033[0m";

    num_correct = 0
    num_omitted = 0;
    unscaled_score = 0.0

    # For every guess/gold answer pair, we keep track if we omit, get the answer correct
    # or get the answer wrong
    for guess, answer in guess_answer_pairs:

        # If we get the answer right, we adjust our counts based on SAT scoring guidelines
        if guess == answer:
            num_correct += 1
            unscaled_score += 1.0

        # If we choose to omit the answer (by returing None or -1) we adjust our counts
        # based on the SAT scoring guidelines
        elif guess == None or guess == -1:
            unscaled_score += 0.0
            num_omitted += 1;

        # If we guess, but we get the answer wrong, we adjust our counts based on the
        # SAT scoring guidelines
        else:
            unscaled_score -= 0.25

    # We then scale our score
    scaled_score = scale_score(unscaled_score, len(guess_answer_pairs))
    score = SCORE_CONVERSION_TABLE[scaled_score]

    # Print out the results of our guesses
    if(verbose):
        print "SAT SCORE: " + str(score)
        print "\tBreakdown:"
        print "\tNumber of Correct Answers: " + str(num_correct) + "/" + str(len(guess_answer_pairs) - num_omitted)
        print "\tNumber of Omitted Answers: " + str(num_omitted) + "/" + str(len(guess_answer_pairs))
        print "\tUnscaled (Raw) Score: " + str(unscaled_score) + "/" + str(len(guess_answer_pairs))
        print "\tScaled Score: " + str(scaled_score) + "/" + str(len(SCORE_CONVERSION_TABLE)-3)
        print "\tConverted SAT Score: " + str(score) + "/800\n"
    return score

# Used to check if our models can correctly guess a wrong answer. This comes from the idea that
# even though our models may not be picking the right answer, they may be able to identify the wrong
# answer. With the scoring of the SAT, if the model predicts the wrong answer, then random guessing
# does not hurt your score (if you are choosing out of 4).
def score_elimination_model(guess_answer_pairs, verbose=False, modelname="Name Not Given"):
    print '\033[95m Bad-Answer Predicting Model: ' + modelname + "\033[0m";

    num_correct = sum([1 if g == a else 0 for g,a in guess_answer_pairs]);
    num_wrong = sum([1 if g != a else 0 for g,a in guess_answer_pairs]);

    if(verbose):
        print "\tBreakdown:"
        print "\tNumber of Correctly Identified Bad Answers: " + str(num_wrong) + "/" + str(len(guess_answer_pairs))
        print "\tPercent Correctly Identified Bad Answers: " + str(float(int(float(num_wrong)/len(guess_answer_pairs)*10000))/100) + "%"
        print "\tNumber of Accidental Right Answers: " + str(num_correct) + "/" + str(len(guess_answer_pairs))
        print "\tPercent Accidental Right Answers: " + str(float(int(float(num_correct)/len(guess_answer_pairs)*10000))/100) + "%\n"

# Scale the score to 67 questions, as if all the questions of the type answered by our models 
# comprised the entire Critical Reading section of the SAT.
def scale_score(unscaled_score, num_questions):
    float_scaled_score = (unscaled_score/num_questions)
    float_scaled_score *= (len(SCORE_CONVERSION_TABLE)-3) 
    return max(int(round(float_scaled_score)), -2)
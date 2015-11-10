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

def score_model(guess_answer_pairs, verbose=False, modelname="Name Not Given"):
    print '\033[95mModel: ' + modelname + "\033[0m";

    num_correct = 0
    unscaled_score = 0.0
    for guess, answer in guess_answer_pairs:
        if guess == answer:
            num_correct += 1
            unscaled_score += 1.0
        elif guess == None or guess == -1:
            unscaled_score += 0.0
        else:
            unscaled_score -= 0.25
    scaled_score = scale_score(unscaled_score, len(guess_answer_pairs))
    score = SCORE_CONVERSION_TABLE[scaled_score]
    if(verbose):
        print "SAT SCORE: " + str(score)
        print "\tBreakdown:"
        print "\tNumber of Correct Answers: " + str(num_correct) + "/" + str(len(guess_answer_pairs))
        print "\tUnscaled (Raw) Score: " + str(unscaled_score) + "/" + str(len(guess_answer_pairs))
        print "\tScaled Score: " + str(scaled_score) + "/" + str(len(SCORE_CONVERSION_TABLE)-3)
        print "\tConverted SAT Score: " + str(score) + "/800\n"
    return score

def score_elimination_model(guess_answer_pairs, verbose=False, modelname="Name Not Given"):
    print '\033[95m Bad-Answer PredictingModel: ' + modelname + "\033[0m";
    num_correct = sum([1 if g == a else 0 for g,a in guess_answer_pairs]);
    num_wrong = sum([1 if g != a else 0 for g,a in guess_answer_pairs]);

    if(verbose):
        print "\tBreakdown:"
        print "\tNumber of Correctly Identified Bad Answers: " + str(num_wrong) + "/" + str(len(guess_answer_pairs))
        print "\tPercent Correctly Identified Bad Answers: " + str(float(int(float(num_wrong)/len(guess_answer_pairs)*10000))/100) + "%"
        print "\tNumber of Accidental Right Answers: " + str(num_correct) + "/" + str(len(guess_answer_pairs))
        print "\tPercent Accidental Right Answers: " + str(float(int(float(num_correct)/len(guess_answer_pairs)*10000))/100) + "%\n"

def scale_score(unscaled_score, num_questions):
    float_scaled_score = (unscaled_score/num_questions)
    float_scaled_score *= (len(SCORE_CONVERSION_TABLE)-3) 
    return max(int(round(float_scaled_score)), -2)

# Sentiment Analysis Module

import nltk
# this we will use to shuffle our dataset, coz for now it is higlhy ordered, first all negative, than positive than neutaral.
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize

from nltk.classify.scikitlearn import SklearnClassifier
# this is basically an nltk module but it is actually a wrapper around the sklearn classifier to add its functionilities in nltk

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC

# combining algos with a vote
# this is so we can inherit from the "nltk classifier" class
from nltk.classify import ClassifierI
# this is how we are gonna choose who got the most votes, we're just gonna take the mode
from statistics import mode
# The mode of a set of data values is the value that appears most often.

import pickle


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        # how many occurences of most popular votes in that list
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)  # this is give us certainity

        return conf


short_pos = open('short_reviews/positive.txt', 'r').read()
short_neg = open('short_reviews/negative.txt', 'r').read()


all_words = []
documents = []

# j is adjective, r is adverb, and v is verb
# allowed_word_types = ['J', 'R', 'V']

allowed_word_types = ['J']  # play with this, you can add 'R' and 'V' too
for p in short_pos.split('\n'):
    documents.append((p, 'pos'))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:  # each w will be a tuple -> (word, pos_tag)
        # w[1] is the pos_tag and w[1][0] is the first letter of that pos_tag
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
for p in short_neg.split('\n'):
    documents.append((p, 'neg'))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:  # each w will be a tuple -> (word, pos_tag)
        # w[1] is the pos_tag and w[1][0] is the first letter of that pos_tag
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# getting pickled documents
documents_f = open('pickled_algos/documents.pickle', 'rb')
documents = pickle.load(documents_f)
documents_f.close()

# _what we're going to end up having to do is take all of these words though literally take every word in every review and compile them and then basically we do is we'll take that list of words and we'll find the most popular words  used and then we take of those most popular words which one appear in positive text and which ones negative text and then we simply just search for those words and whichever one it has more negative words or more positive words that's how we classify it_

all_words = nltk.FreqDist(all_words)

# getting pickled word_features
word_features5k_f = open('pickled_algos/word_features5k.pickle', 'rb')
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in words:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)


train_set = featuresets[:10000]
test_set = featuresets[10000:]


# ### Naive Byes Algorithm
# posterior = prior occurences * occurences / likelihood

open_file = open('pickled_algos/originalnaivebayes5k.pickle', 'rb')
classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/MNB_classifier5k.pickle', 'rb')
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/BNB_classifier5k.pickle', 'rb')
BNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/LogisticRegression_classifier5k.pickle', 'rb')
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/LinearSVC_classifier5k.pickle', 'rb')
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/SGDC_classifier5k.pickle', 'rb')
SGDC_classifier = pickle.load(open_file)
open_file.close()

# ### Remember
#
# >These all above algos(classifiers) have got their own parameters, which we can tweak and imrove our accuracy, currently here we have used the default values for each of them.
# >
# >If you want to know more about those params, you can check those out on sklearn website.

voted_classifier = VoteClassifier(
    classifier,
    MNB_classifier,
    BNB_classifier,
    # SGDClassifier_classifier,
    LinearSVC_classifier,
    # NuSVC_classifier,
    LogisticRegression_classifier)


def sentiment(text):
    feat = find_features(text)

    return voted_classifier.classify(feat), voted_classifier.confidence(feat)

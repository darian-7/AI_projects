# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023

# Darian Irani - 16 Sept 2023

"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
from itertools import chain
from itertools import islice

'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior:Modif {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.05, bigram_laplace=0.004, bigram_lambda=0.4, pos_prior=0.75, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # Training
    pos_review = [review for review, label in zip(train_set, train_labels) if label == 1]
    neg_review = [review for review, label in zip(train_set, train_labels) if label != 1]

    pos_word_count = Counter(chain.from_iterable(pos_review))
    neg_word_count = Counter(chain.from_iterable(neg_review))

    
    pos_bigrams = [bigram for review in pos_review for bigram in zip(review, review[1:])]
    pos_bigram_counter = Counter(pos_bigrams)

    neg_bigrams = [bigram for review in neg_review for bigram in zip(review, review[1:])]
    neg_bigram_counter = Counter(neg_bigrams)

    total_pos_words = sum(pos_word_count.values())
    total_neg_words = sum(neg_word_count.values())

    total_pos_bigrams = sum(pos_bigram_counter.values())
    total_neg_bigrams = sum(neg_bigram_counter.values())

    # Development
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        doc_bigrams = list(zip(doc, doc[1:]))

        # Unigram
        uni_pos_prob = math.log(pos_prior)
        uni_neg_prob = math.log(1 - pos_prior)
        
        for word in doc:
            if word in pos_word_count:
                uni_pos_prob += math.log((pos_word_count[word] + unigram_laplace) / (total_pos_words + unigram_laplace * (len(pos_word_count)+1)))
            else:
                uni_pos_prob += math.log(unigram_laplace / (total_pos_words + unigram_laplace * (len(pos_word_count)+1)))
        
            if word in neg_word_count:
                uni_neg_prob += math.log((neg_word_count[word] + unigram_laplace) / (total_neg_words + unigram_laplace * (len(neg_word_count)+1)))
            else:
                uni_neg_prob += math.log(unigram_laplace / (total_neg_words + unigram_laplace * (len(neg_word_count)+1)))

        # Bigram
        bi_pos_prob = math.log(pos_prior)
        bi_neg_prob = math.log(1 - pos_prior)
        
        for bigram in doc_bigrams:
            if bigram in pos_bigram_counter:
                bi_pos_prob += math.log((pos_bigram_counter[bigram] + bigram_laplace) / (total_pos_bigrams + bigram_laplace * (len(pos_word_count)+1)))
            else:
                bi_pos_prob += math.log(bigram_laplace / (total_pos_bigrams + bigram_laplace * (len(pos_word_count)+1)))
        
            if bigram in neg_bigram_counter:
                bi_neg_prob += math.log((neg_bigram_counter[bigram] + bigram_laplace) / (total_neg_bigrams + bigram_laplace * (len(neg_word_count)+1)))
            else:
                bi_neg_prob += math.log(bigram_laplace / (total_neg_bigrams + bigram_laplace * (len(neg_word_count)+1)))
                
        # Mixture model
        pos_prob = bigram_lambda * bi_pos_prob + (1 - bigram_lambda) * uni_pos_prob     
        neg_prob = bigram_lambda * bi_neg_prob + (1 - bigram_lambda) * uni_neg_prob
        
        yhats.append(pos_prob > neg_prob)

    return yhats






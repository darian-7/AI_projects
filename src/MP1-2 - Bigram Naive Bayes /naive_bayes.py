# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023

# Darian Irani - 10 Sept 2023

import reader
import math
from tqdm import tqdm
from collections import Counter
from itertools import chain


def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir, testdir, stemming, lowercase, silently)
    return train_set, train_labels, dev_set, dev_labels

def naiveBayes(dev_set, train_set, train_labels, laplace=0.05, pos_prior=0.75, silently=False):
    print_values(laplace, pos_prior)
    
# Training 
    pos_review = [review for review, label in zip(train_set, train_labels) if label == 1]
    neg_review = [review for review, label in zip(train_set, train_labels) if label != 1]

    pos_word_count = Counter(chain.from_iterable(pos_review))
    neg_word_count = Counter(chain.from_iterable(neg_review))

    total_pos_words = sum(pos_word_count.values())
    total_neg_words = sum(neg_word_count.values())

    vocab_size = len(set(pos_word_count) | set(neg_word_count))

# Development
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1 - pos_prior)
        
        for word in doc:
            if word in pos_word_count:
                pos_prob += math.log((pos_word_count[word] + laplace) / (total_pos_words + laplace * (len(pos_word_count.keys())+1)))
            else:
                pos_prob += math.log((laplace) / (total_pos_words + laplace * (len(pos_word_count.keys())+1)))
        
            if word in neg_word_count:
                neg_prob += math.log((neg_word_count[word] + laplace) / (total_neg_words + laplace * (len(neg_word_count.keys())+1)))
            else:
                neg_prob += math.log((laplace) / (total_neg_words + laplace * (len(neg_word_count.keys())+1)))

        if pos_prob > neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)
            
    return yhats



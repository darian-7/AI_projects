"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Constants
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """

    # Initialization to store dicts and counts
    init_prob = defaultdict(float)  # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(float)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(float))    # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    total_tags = defaultdict(int)
    tag_pairs = defaultdict(lambda: defaultdict(int))
    tag_word_pairs = defaultdict(lambda: defaultdict(int))
    
    # Count freq of tags, tag pairs, and tag-word pairs
    for sentence in sentences:
        prev_tag = None
        for word, tag in sentence:
            total_tags[tag] += 1
            tag_word_pairs[tag][word] += 1
            if prev_tag is not None:
                tag_pairs[prev_tag][tag] += 1
            prev_tag = tag
    
    total_tags_count = sum(total_tags.values())
    for tag, count in total_tags.items():
        init_prob[tag] = count / total_tags_count   # Initial tag prob
        for word, word_count in tag_word_pairs[tag].items():    # Given a tag, calc e_prob for each word
            emit_prob[tag][word] = word_count / count

        # Smoothing for unseen words given a tag
        total_word_count_for_tag = sum(tag_word_pairs[tag].values())
        emit_prob[tag]['UNK'] = emit_epsilon / (total_word_count_for_tag + emit_epsilon)

        # Transition prob from one tag to another
        for next_tag, next_tag_count in tag_pairs[tag].items():
            trans_prob[tag][next_tag] = next_tag_count / count

    return init_prob, emit_prob, trans_prob
        

def viterbi_stepforward(i, word, prev_probs, tag_sequences, emit_probs, trans_probs):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    curr_probs = {} # This should store the log_prob for all the tags at current column (i)
    curr_sequences = {} # This should store the tag sequence to reach each tag at column (i)
    
    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    tags = emit_probs.keys()    # Get tags from e_prob
    
    if i == 0:  # First word in sentence
        for tag in tags:
            curr_probs[tag] = prev_probs.get(tag, math.log(1e-10))  # Initial prob of each tag
            if word in emit_probs[tag]:
                curr_probs[tag] += math.log(emit_probs[tag][word])  # Prob of emitting current word given current tag, if word not in training data then use UNK token
            else:
                curr_probs[tag] += math.log(emit_probs[tag].get('UNK', 1e-10))
            curr_sequences[tag] = [tag]
    else:
        for curr_tag in tags:   # Consider all possible previous tags
            max_prob = float('-inf')
            max_seq = []
            for prev_tag in tags:
                prob = prev_probs[prev_tag] + math.log(trans_probs[prev_tag].get(curr_tag, 1e-10))  # Prob of transitioning from previous tag to current
                if word in emit_probs[curr_tag]:    # Add prob of emitting current word given current tag
                    prob += math.log(emit_probs[curr_tag][word])
                else:
                    prob += math.log(emit_probs[curr_tag].get('UNK', 1e-10))
                
                if prob > max_prob:
                    max_prob = prob  # Update max value if its the best seq for this tag
                    max_seq = tag_sequences[prev_tag] + [curr_tag]
            
            curr_probs[curr_tag] = max_prob # Store this best prob and seq for the tag
            curr_sequences[curr_tag] = max_seq

    return curr_probs, curr_sequences


def viterbi_1(train, test, get_probs_func = training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_probs, emit_probs, trans_probs = get_probs_func(train)
    all_predictions = []
    
    for sentence in test:
        log_probs = {tag: math.log(init_probs.get(tag, 1e-10)) for tag in emit_probs}
        tag_seqs = {tag: [] for tag in emit_probs}
        
        for idx, word in enumerate(sentence):
            log_probs, tag_seqs = viterbi_stepforward(idx, word, log_probs, tag_seqs, emit_probs, trans_probs)
        
        # Get the most probbable tag sequence for the last word
        final_tags = max(log_probs, key=log_probs.get)
        best_sequence = [(word, tag) for word, tag in zip(sentence, tag_seqs[final_tags])]
        
        all_predictions.append(best_sequence)
    
    return all_predictions



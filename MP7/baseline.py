from collections import defaultdict, Counter

"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    word_tag_dict = defaultdict(Counter)
    overall_tag_freq = Counter()

    # Process training data
    for sentence in train:
        for word, tag in sentence:
            word_tag_dict[word][tag] += 1
            overall_tag_freq[tag] += 1

    # most_common() for efficiency.
    most_freq_tags = {word: tags.most_common(1)[0][0] for word, tags in word_tag_dict.items()}
    most_frequent_tag = overall_tag_freq.most_common(1)[0][0]

    # Process test data
    tagged = [[(word, most_freq_tags.get(word, most_frequent_tag)) for word in sentence] for sentence in test]

    return tagged
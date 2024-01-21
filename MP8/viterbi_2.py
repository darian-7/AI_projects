"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    all_words = set([word for sentence in train for word, _ in sentence])

    # Hapax words and their tags
    word_freqs = {}
    for sentence in train:
        for word, tag in sentence:
            word_freqs[word] = word_freqs.get(word, 0) + 1
            
    hapax_words = {word for word, count in word_freqs.items() if count == 1}
    
    # Tag prob in hapax words
    hapax_tags = [tag for sentence in train for word, tag in sentence if word in hapax_words]
    total_hapax_tags = len(hapax_tags)
    hapax_tag_probs = {tag: hapax_tags.count(tag) / total_hapax_tags for tag in set(hapax_tags)}

    # Compute set of all tags in training data
    all_tags = set([tag for sentence in train for _, tag in sentence])

    # Calculate transition probs
    transitions = {}
    for sentence in train:
        for i in range(1, len(sentence)):
            prev_tag = sentence[i-1][1]
            curr_tag = sentence[i][1]
            transitions[(prev_tag, curr_tag)] = transitions.get((prev_tag, curr_tag), 0) + 1

    tag_counts = {tag: 0 for tag in all_tags}
    for sentence in train:
        for _, tag in sentence:
            tag_counts[tag] += 1

    transition_probs = {}
    for (prev_tag, curr_tag), count in transitions.items():
        transition_probs[(prev_tag, curr_tag)] = count / tag_counts[prev_tag]

    # Emission probs calculation
    alpha = 1e-5

    emission_counts = {(word, tag): 0 for word in all_words for tag in all_tags}
    for sentence in train:
        for word, tag in sentence:
            emission_counts[(word, tag)] += 1

    emission_probs = {}
    for (word, tag), count in emission_counts.items():
        emission_probs[(word, tag)] = (count + alpha * hapax_tag_probs.get(tag, 1)) / (tag_counts[tag] + len(all_words) * alpha * hapax_tag_probs.get(tag, 1))

    # Viterbi Algorithm
    tagged_sentences = []

    for sentence in test:
        dp = [{tag: (-float('inf'), None) for tag in all_tags} for _ in sentence]
        for tag in all_tags:
            word = sentence[0]
            emission = emission_probs.get((word, tag), alpha * hapax_tag_probs.get(tag, 1))
            dp[0][tag] = (emission, None)

        for i in range(1, len(sentence)):
            for tag in all_tags:
                word = sentence[i]
                emission = emission_probs.get((word, tag), alpha * hapax_tag_probs.get(tag, 1))
                max_prob, best_prev_tag = max((dp[i-1][prev_tag][0] * transition_probs.get((prev_tag, tag), 0) * emission, prev_tag) for prev_tag in all_tags)
                dp[i][tag] = (max_prob, best_prev_tag)

        # Backtrack to find the best path
        _, last_best_tag = max((value[0], tag) for tag, value in dp[-1].items())
        best_path = [last_best_tag]

        for i in range(len(sentence)-1, 0, -1):
            best_prev_tag = dp[i][best_path[-1]][1]
            best_path.append(best_prev_tag)

        best_path = list(reversed(best_path))
        tagged_sentences.append(list(zip(sentence, best_path)))

    return tagged_sentences
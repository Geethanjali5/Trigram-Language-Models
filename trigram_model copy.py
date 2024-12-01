import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    # Add padding with (n - 1) 'START' tokens and one 'STOP' token
    padded_sequence = ['START'] * (n - 1) + sequence + ['STOP']
    
    # Special case for unigrams to include the 'START' token
    if n == 1:
        padded_sequence = ['START'] + sequence + ['STOP']
    
    # Extracting n-grams as tuples
    ngrams = [tuple(padded_sequence[i:i + n])
              for i in range(len(padded_sequence) - n + 1)]
    
    return ngrams

    return []


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        # Initialize dictionaries for unigrams, bigrams, and trigrams
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.tokencount = 0
        ##Your code here
        #Iterate through each sentence in the corpus
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)
            
            # Count each n-gram
            for unigram in unigrams:
                self.unigramcounts[unigram] += 1
                self.tokencount += 1
            for bigram in bigrams:
                self.bigramcounts[bigram] += 1
            for trigram in trigrams:
                self.trigramcounts[trigram] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        count_trigram = self.trigramcounts[trigram]
        count_bigram_word = self.bigramcounts[tuple([trigram[0],trigram[1]])]
        return count_trigram / count_bigram_word if count_bigram_word > 0 else 1/len(self.lexicon)

        

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        count_bigram = self.bigramcounts[bigram]
        count_first_word = self.unigramcounts[tuple([bigram[0],])]
        # Calculate the probability, default to 0 if the denominator is 0
        return count_bigram / count_first_word


    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        
        # Calculate the probability, default to 0 if the denominator is 0
        return self.unigramcounts[unigram] / self.tokencount

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  

    def generate_sentence(self, t=20): 
       """
       Generate a random sentence from the trigram model. 
       t specifies the max length, but the sentence may be shorter if STOP is reached.
       """
       # Initialize the sentence with the first two 'START' tokens
       sentence = ['START', 'START']
    
       # Continue generating words until 'STOP' is reached or the max length t is reached
       while len(sentence) < t + 2:  # +2 because 'START', 'START' are already in the list
        # Get the last two words as context
        context = tuple(sentence[-2:])
        
        possible_next_words = []
        possible_next_counts = []
             
        for trigram, count in self.trigramcounts.items():
            if trigram[:2] == context:
                possible_next_words.append(trigram[2])
                possible_next_counts.append(count)
        
        if not possible_next_words:
            # If there are no possible next words, stop the sentence generation
            break

        probabilities = np.array(possible_next_counts) / sum(possible_next_counts)
        
        next_word = np.random.choice(possible_next_words, p=probabilities)

        sentence.append(next_word)

        if next_word == 'STOP':
            break
    
       # Return the generated sentence as a list of words (excluding 'START' and 'STOP' tokens)
       return sentence[2:]  

              
    
    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        # Interpolation weights
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        
        raw_trigram_prob = self.raw_trigram_probability(trigram)
        raw_bigram_prob = self.raw_bigram_probability((trigram[1], trigram[2]))
        raw_unigram_prob = self.raw_unigram_probability((trigram[2],))
    
        # Calculate the smoothed probability using linear interpolation
        smoothed_prob = (lambda1 * raw_trigram_prob) + (lambda2 * raw_bigram_prob) + (lambda3 * raw_unigram_prob)
    
        return smoothed_prob

        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        # Initialize the total log probability
        total_log_prob = 0.0

        trigrams = get_ngrams(sentence, 3)

        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram)

            if prob > 0:
                # Convert to log space using math.log2 and add to total log probability
                total_log_prob += math.log2(prob)
            else:
                # If the probability is zero, return negative infinity
             return float("-inf")

        return total_log_prob

    def perplexity(self, corpus):
    
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total_log_prob = 0.0
        total_word_count = 0
    
        # Iterate through each sentence in the corpus
        for sentence in corpus:
            sentence_log_prob = self.sentence_logprob(sentence)
        
            total_log_prob += sentence_log_prob

            total_word_count += len(sentence)  # Each sentence includes all tokens in it
    
        # Calculate the average log probability
        if total_word_count > 0:
            avg_log_prob = total_log_prob / total_word_count
            # Compute perplexity using the formula 2^(-average_log_prob)
            perplexity_value = 2 ** (-avg_log_prob)
            return perplexity_value
    
        return float("inf")

        
def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    """
    Trains two trigram models using training_file1 and training_file2, then tests them
    on essays in testdir1 and testdir2, and returns the classification accuracy.
    
    Arguments:
    training_file1 -- path to training data for high skill essays
    training_file2 -- path to training data for low skill essays
    testdir1 -- directory containing test essays for high skill
    testdir2 -- directory containing test essays for low skill
    
    Returns:
    Accuracy of the classification as a float between 0 and 1
    """
    
    # Train the trigram models
    model1 = TrigramModel(training_file1)  # High skill model
    model2 = TrigramModel(training_file2)  # Low skill model

    total = 0  # Total number of test essays
    correct = 0  # Number of correctly classified essays

    # Test the high skill essays (testdir1)
    for f in os.listdir(testdir1):
        filepath = os.path.join(testdir1, f)
        test_corpus = corpus_reader(filepath, model1.lexicon)
        
        # Calculate perplexity for both models
        pp_model1 = model1.perplexity(test_corpus)
        test_corpus = corpus_reader(filepath, model2.lexicon)  # Reload the corpus iterator
        pp_model2 = model2.perplexity(test_corpus)
        
        total += 1
        
        # Model1 should have lower perplexity for high skill essays
        if pp_model1 < pp_model2:
            correct += 1

    # Test the low skill essays (testdir2)
    for f in os.listdir(testdir2):
        filepath = os.path.join(testdir2, f)
        test_corpus = corpus_reader(filepath, model1.lexicon)
        
        # Calculate perplexity for both models
        pp_model1 = model1.perplexity(test_corpus)
        test_corpus = corpus_reader(filepath, model2.lexicon)  # Reload the corpus iterator
        pp_model2 = model2.perplexity(test_corpus)
        
        total += 1
        
        # Model2 should have lower perplexity for low skill essays
        if pp_model2 < pp_model1:
            correct += 1

    # Calculate accuracy
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    # Check if an argument was provided
    if len(sys.argv) > 1:
        corpus_file = sys.argv[1]
    else:
        # Use a default path if no argument is given
        corpus_file = '/Users/geethanjali.p/Downloads/hw1_data/brown_train.txt'  # Replace this with your actual file path

    # Initialize the TrigramModel with the specified or default corpus file
    model = TrigramModel(corpus_file)

    # Test case for Step 1 - Extracting n-grams
    print("\n=== Step 1: Extracting n-grams from a sentence ===")
    test_sequence = ["natural", "language", "processing"]

    for n in range(1, 3):
        print(f"\nTesting get_ngrams with n={n} for sequence: {test_sequence}")
        result = get_ngrams(test_sequence, n)
        print(result)

    # === Test case for Step 2: Counting n-grams in a corpus ===
    print("\n=== Step 2: Counting n-grams in the corpus ===")
    
    # Sample output for a few n-grams
    sample_unigram = ('the',)
    sample_bigram = ('START', 'the')
    sample_trigram = ('START', 'START', 'the')

   
    print(f"Count for trigram {sample_trigram}: {model.trigramcounts.get(sample_trigram, 'Not found')}")
    print(f"Count for bigram {sample_bigram}: {model.bigramcounts.get(sample_bigram, 'Not found')}")
    print(f"Count for unigram {sample_unigram}: {model.unigramcounts.get(sample_unigram, 'Not found')}")


    # Test case for Step 3 - Raw probabilities

    print("\n=== Step 3: Raw n-gram probabilities ===")
    
    # Test raw_unigram_probability
    unigram_test = ('the',)
    print(f"\nTesting raw_unigram_probability for {unigram_test}:")
    print("Probability:", model.raw_unigram_probability(unigram_test))

    # Test raw_bigram_probability
    bigram_test = ('START', 'the')
    print(f"\nTesting raw_bigram_probability for {bigram_test}:")
    print("Probability:", model.raw_bigram_probability(bigram_test))

    # Test raw_trigram_probability
    trigram_test = ('START', 'START', 'the')
    print(f"\nTesting raw_trigram_probability for {trigram_test}:")
    print("Probability:", model.raw_trigram_probability(trigram_test))

    # Test edge case with an unseen trigram
    unseen_trigram_test = ('unknown1', 'unknown2', 'unknown3')
    print(f"\nTesting raw_trigram_probability for an unseen trigram {unseen_trigram_test}:")
    print("Probability:", model.raw_trigram_probability(unseen_trigram_test))

    # Test case for Step 4 - Smoothed trigram probability
    print("\n=== Step 4: Smoothed trigram probability ===")
    trigram_test = ('START', 'START', 'the')
    print(f"\nTesting smoothed_trigram_probability for {trigram_test}:")
    print("Smoothed Probability:", model.smoothed_trigram_probability(trigram_test))



     # Test case for Step 5 - Sentence Log Probability
    test_sentence = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    log_probability = model.sentence_logprob(test_sentence)
    
    print("\n=== Step 5: Sentence Log Probability ===")
    print(f"Log Probability of the sentence '{' '.join(test_sentence)}': {log_probability}")

    # Testing perplexity on the Brown corpus test file
    print("\n=== Step 6: Testing Perplexity on the Brown corpus ===")
    
    # Specify the test file path for the Brown corpus
    test_corpus_file = '/Users/geethanjali.p/Downloads/hw1_data/brown_test.txt'  # Replace with the path to your test file

    # Use corpus_reader to read the test corpus with the model's lexicon
    test_corpus = corpus_reader(test_corpus_file, model.lexicon)

    # Calculate and print the perplexity
    perplexity_value = model.perplexity(test_corpus)
    print(f"Perplexity of the test corpus: {perplexity_value}")



     # Test case for Step 7 - Essay Scoring Experiment
    print("\n=== Step 7: Essay Scoring Experiment ===")

     # Define the paths for training and testing files/directories
    training_file1 = '/Users/geethanjali.p/Downloads/hw1_data/ets_toefl_data/train_high.txt'  # High skill training data
    training_file2 = '/Users/geethanjali.p/Downloads/hw1_data/ets_toefl_data/train_low.txt'   # Low skill training data
    testdir1 = '/Users/geethanjali.p/Downloads/hw1_data/ets_toefl_data/test_high'             # High skill test essays
    testdir2 = '/Users/geethanjali.p/Downloads/hw1_data/ets_toefl_data/test_low'              # Low skill test essays

    # Call the essay_scoring_experiment function
    accuracy = essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2)

    # Print the accuracy
    print(f"Essay Scoring Accuracy: {accuracy * 100:.2f}%")

# put test code here...
# or run the script from the command line with
# $ python -i trigram_model.py [corpus_file]
# >>>
#
# you can then call methods on the model instance in the interactive
# Python prompt.
# Testing perplexity:
# dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
# pp = model.perplexity(dev_corpus)
# print(pp)
# Essay scoring experiment:
# acc = essay_scoring_experiment('train_high.txt', 'train_low.txt","test_high", "test_low")
# print(acc)


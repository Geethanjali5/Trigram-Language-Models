# **Trigram Language Model**

## **Overview**
This project implements an **N-gram Language Model**, focusing on **trigram probabilities** for text processing. The model can:
- Extract **unigrams, bigrams, and trigrams** from a corpus.
- Compute **raw and smoothed probabilities** for trigrams.
- **Generate sentences** based on trigram probabilities.
- Compute **log probabilities** and **perplexity** to evaluate text quality.
- Perform **essay classification** based on language complexity.

## **Features**
- **Corpus Processing**: Reads and preprocesses text, handling unknown words.
- **N-gram Extraction**: Generates **unigrams, bigrams, and trigrams** with appropriate padding.
- **Probability Estimation**:
  - Raw **unigram, bigram, and trigram probabilities**.
  - **Smoothed trigram probabilities** using linear interpolation.
- **Sentence Generation**: Generates a random sentence using trigram probabilities.
- **Perplexity Calculation**: Measures how well the model predicts test data.
- **Essay Classification**: Trains two trigram models (**high vs. low skill essays**) and classifies new essays based on perplexity scores.

## **Installation**
Ensure you have Python installed. No additional dependencies are required.
```sh
python -m venv env
source env/bin/activate  # On Mac/Linux
env\Scripts\activate  # On Windows
```

"""
Code for computing TAACO givenness scores.
"""
import math
from stanza.models.common.doc import Sentence




def giv_avg_entire_text(text : list[Sentence]) -> float:
    """Calculate the global givenness average for the entire text."""

    current_lemmas = set()
    total_lemmas = 0
    repeated_lemmas = 0
    pronouns = 0
    global_givenness = 0
    content_tags = ["NN", "VB", "PN", "NOUN", "VERB", "PRON"]
    punctuation_marks = ["MAD", "MID", "PAD", "PUNCT"]
    swedish_pronouns = "han hon hans hennes de dem deras mig dig vi ni dess du jag den det vi ni".split(" ")

    for sentence in text:
        for word in sentence.words:
            
            try:

                lemma = word.lemma
                upos = word.upos             # upos = Universal Part Of Speech Tag
                
                current_lemmas.add(lemma)           # save number of unique lemmas

                # if the word is a noun, verb or pronoun and lemma has been repeated once
                if lemma in current_lemmas and upos in content_tags:
                    repeated_lemmas += 1
                
                # if the word is a pronoun and exists in the defined list of swedish pronouns
                if lemma in swedish_pronouns and upos == "PRON":
                    pronouns += 1
                
                # if word is not a punctuation mark
                if upos not in punctuation_marks:
                    total_lemmas += 1
            
            except:
                continue
    
    # Calculate global givenness according to the formula
    global_givenness = (repeated_lemmas + pronouns) / total_lemmas
    
    return global_givenness


def giv_avg(text : list[Sentence]) -> float:
    """Calculate the global TAACO givenness average by sentence instead of the entire text."""

    current_lemmas = set()
    global_givenness = 0
    total_sentences = 0
    total_sentence_givenness = 0
    content_tags = ["NN", "VB", "PN", "NOUN", "VERB", "PRON"]
    punctuation_marks = ["MAD", "MID", "PAD", "PUNCT"]
    swedish_pronouns = "han hon hans hennes de dem deras mig dig vi ni dess du jag den det vi ni".split(" ")

    for sentence in text:

        # In this function, the metrics used to calculate givenness are
        # reset in each iteration of the sentence loop so the givenness
        # is calculated for each sentence by itself instead.

        total_lemmas = 0
        repeated_lemmas = 0
        pronouns = 0

        for word in sentence.words:
            
            try:
                lemma = word.lemma
                upos = word.upos               # upos = Universal Part Of Speech Tag
                
                
                           # save number of unique lemmas

                # if the word is a noun, verb or pronoun and lemma has been repeated once
                if lemma in current_lemmas and upos in content_tags:
                    repeated_lemmas += 1
                
                # if the word is a pronoun and exists in the defined list of swedish pronouns
                if lemma in swedish_pronouns and upos == "PRON":
                    pronouns += 1
                
                # if word is not a punctuation mark
                if upos not in punctuation_marks:
                    total_lemmas += 1

                current_lemmas.add(lemma)

            except:
                continue
        
        total_sentence_givenness += (repeated_lemmas + pronouns) / total_lemmas
        total_sentences += 1
    
    # Calculate global givenness according to the formula
    global_givenness = total_sentence_givenness / total_sentences
    
    return global_givenness


def giv_stdev(text : list[Sentence]) -> float:
    """Calculate the standard deviation of givenness with respect to each sentence in the text."""

    current_lemmas = set()
    separate_sentence_givenness = []
    content_tags = ["NN", "VB", "PN", "NOUN", "VERB", "PRON"]
    punctuation_marks = ["MAD", "MID", "PAD", "PUNCT"]
    swedish_pronouns = "han hon hans hennes de dem deras mig dig vi ni dess du jag den det vi ni".split(" ")

    for sentence in text:

        # In this function, the metrics used to calculate givenness are
        # reset in each iteration of the sentence loop so the givenness
        # is calculated for each sentence by itself instead.

        total_lemmas = 0
        repeated_lemmas = 0
        pronouns = 0

        for word in sentence.words:
            
            try:

                lemma = word.lemma
                upos = word.upos               # upos = Universal Part Of Speech Tag
                
                
                          # save number of unique lemmas

                # if the word is a noun, verb or pronoun and lemma has been repeated once
                if lemma in current_lemmas and upos in content_tags:
                    repeated_lemmas += 1
                
                # if the word is a pronoun and exists in the defined list of swedish pronouns
                if lemma in swedish_pronouns and upos == "PRON":
                    pronouns += 1
                
                # if word is not a punctuation mark
                if upos not in punctuation_marks:
                    total_lemmas += 1

                current_lemmas.add(lemma) 

            except:
                continue
        
        separate_sentence_givenness.append((repeated_lemmas + pronouns) / total_lemmas)
    
    # Calculate standard deviation of givenness

    return stdev(separate_sentence_givenness)


def variance(data, ddof=0):
    """Helper function that calculates variance"""
    
    total_datapoints = len(data)
    mean = sum(data) / total_datapoints
    return sum((x - mean) ** 2 for x in data) / (total_datapoints - ddof)


def stdev(data):
    """Helper function that calculates standard deviation."""
    
    var = variance(data)
    std_dev = math.sqrt(var)
    return std_dev
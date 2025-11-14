from itertools import permutations
from cached_SBERT import CachedSBERTVectorizer
from text_scorer import TextScorer
from stanza.models.common.doc import Sentence


def brute_force_search(sentences : list[Sentence], \
                       scorer: TextScorer) -> list[Sentence]:
    """
    Find the global optimum. 
    Works well on problems with less than 8 sentences.
    :param sentences: a list of sentences to be ordered.
    :param scorer: a scoring function that compares orderings.
    :return: a list of sentences in the optimal order.
    """
    print("Bruteforce")
    for sentence_order in permutations(sentences):
        order_score = scorer.compute_final_score(sentence_order)
        best_score = 0.0
        if order_score > best_score:
            best_score = order_score
            best_order = sentence_order

    return best_order
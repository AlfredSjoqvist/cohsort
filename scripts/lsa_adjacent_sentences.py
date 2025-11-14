
# LSA sentence adjacent: LSASS1(index 40)
# This index computes mean LSA cosines for adjacent, sentence-to-sentence (abbreviated as "ass") units. 
# This measures how conceptually similar each sentence is to the next sentence. 

# LSASS1d (index 41)
# This index computes standard deviation of LSA cosines for adjacent, sentence-to-sentence (abbreviated as "ass") units. 
# This measures how consistent adjacent sentences are overlaped semantically.

from SBERT import SBERTVectorizer
import cosine_sim
from stanza.models.common.doc import Sentence


class LSAAdjacentSentences:
    """
    Computes the LSASS1 and LSASS1d scores for sentences using
    sentence embeddings.
    """

    def __init__(self, vectorizer: SBERTVectorizer):
        """
        :param vectorizer: the vectorizer for sentence embeddings.
        """
        self.vectorizer = vectorizer

    def lsa_adjacent(self, sentences: list[Sentence]) -> tuple[float, float]:
        """
        Compute the average and standard deviation cosine similarity
        for each adjacent sentence.

        :param sentences: a list of sentences.
        :return: a 2-tuple where (avg_cos_sims, std_cos_sims).
        """

        cos_sims = []
        
        for i, sentence in enumerate(sentences[:-1]):
            next_sentence = sentences[i + 1]
            embedding = self.vectorizer.vectorize(sentence)
            next_embedding = self.vectorizer.vectorize(next_sentence)
            cossim = cosine_sim.cos_sim(embedding, next_embedding)
            cos_sims.append(cossim)

        avg_cos_sims = cosine_sim.norm_avg_cos_sims(cos_sims)
        std_cos_sims = cosine_sim.norm_std_cos_sims(cos_sims)

        return avg_cos_sims, std_cos_sims

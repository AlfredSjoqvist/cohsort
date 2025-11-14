"""
LSA sentence all: LSASSp (index 42)
Like LSA sentence adjacent (LSAassa), this index computes mean LSA cosines.
However, for this index all sentence combinations are considered, not just adjacent sentences. 
LSApssa computes how conceptually similar each sentence is to every other sentence in the text.

LSASSpd (index 43)
This index computes the standard deviation of LSA cosine of all sentence pairs within paragraphs.
"""
import parsing
import cosine_sim
from cached_SBERT import CachedSBERTVectorizer
from stanza.models.common.doc import Sentence
from parsing import Parser


class LSAAllSentences():
    """ Computes LSASSp and LSASSpd scores. """

    def __init__(self, vectorizer: CachedSBERTVectorizer):
        """
        :param vectorizer: the vectorizer for sentence embeddings.
        """
        self.vectorizer = vectorizer
    
    def all_cosine_sim(self, sentences: list[Sentence]) -> list[float]: 
        """ 
        Calculate the cosine siularities for all sentence pairs. 
        :param sentences: a list of sentences
        :return: A list of all cosine simularities.
        """
        # Calculate the cos_sum between one vector of the sentence "index" and the sentence "query"
        # Index here is the current sentence, and query is the sentence which is compared against
        # Do this for each index
        cos_sims = [
            cosine_sim.cos_sim(self.vectorizer.vectorize(index), \
                               self.vectorizer.vectorize(query))
            for i, index in enumerate(sentences)
            for j, query in enumerate(sentences)
            if i != j
        ]
        return cos_sims
    
    def average_and_std_dev(self, sentences: list[Sentence]) -> tuple[float, float]:
        """
        Calculate the avarage and standard deviation 
        of all cosine simularities betweens sentence pairs.
        :param sentences: a list of sentences.
        :return: a 2-tuple where (LSASSp, LSASSpd).
        """

        if len(sentences) <= 1:
            return None
        
        # Caclulates the cosine simularities for each sentence pair
        cos_sims = self.all_cosine_sim(sentences)
        # Caclulates the LSASSp and LSASSpd scores    
        LSASSp = cosine_sim.norm_avg_cos_sims(cos_sims)
        LSASSpd = cosine_sim.norm_std_cos_sims(cos_sims)

        return LSASSp, LSASSpd
        

if __name__ == "__main__":    
    parser = Parser()
    summary = parsing.load_summary("baljväxter.txt")
    segmented_summary = parser.parse(summary)
    segmented_summary_text = parsing.to_text(segmented_summary)
    vectorizer = CachedSBERTVectorizer()
    test = LSAAllSentences(vectorizer)
    print(test.average_and_std_dev(segmented_summary_text))





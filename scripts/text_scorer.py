
from lsa_adjacent_sentences import LSAAdjacentSentences
import taaco_givenness
from lsa_givenness import LSAGivenness
from parsing import Parser
from SBERT import SBERTVectorizer
from cached_SBERT import CachedSBERTVectorizer
from stanza.models.common.doc import Sentence
import content_word_overlap
import syntactic_similarity


class TextScorer:
    """
    Scores sequences of sentences using certain Coh-metrix measurements.
    """

    def __init__(self, vectorizer: SBERTVectorizer, weights_values=None):
        """
        :param vectorizer: the vectorizer for sentence embeddings.
        """
        self.vectorizer = vectorizer
        if weights_values:
            self.weights = {
            # "LSASSp_weight": 1.0, # Not sure if this index is affected by the reordering of sentences.
            # "LSASSpd_weight": 1.0, # Not sure if this index is affected by the reordering of sentences.
            "LSASS1_weight": weights_values[0],
            "LSASS1d_weight": weights_values[1],
            "LSAGN_weight": weights_values[2],
            "LSAGNd_weight": weights_values[3],
            "SYNSTRUTa_weight": weights_values[4],
            "CRFCW01_weight": weights_values[5]
        }
        else:
            self.weights = {
            # "LSASSp_weight": 1.0, # Not sure if this index is affected by the reordering of sentences.
            # "LSASSpd_weight": 1.0, # Not sure if this index is affected by the reordering of sentences.
            "LSASS1_weight": 1.0,
            "LSASS1d_weight": 1.0,
            "LSAGN_weight": 1.0,
            "LSAGNd_weight": 1.0,
            "SYNSTRUTa_weight": 1.0,
            "CRFCW01_weight": 1.0
        }
        self.lsa_adjacent = LSAAdjacentSentences(self.vectorizer)
        self.lsa_givenness = LSAGivenness(self.vectorizer)

    def compute_scores(self, sentences: list[Sentence]) -> list[float]:
        """
        Computes the individual scores for the individual metrices.
        :return: a list of all the scores, as floats.
        """
        lsass1, lsass1d = self.lsa_adjacent.lsa_adjacent(sentences)
        lsa_giv, lsa_giv_d = self.lsa_givenness.givenness(sentences)
        synstruta = syntactic_similarity.avg_syntax_similarity(sentences)  # synt.synt_struc_adj(sentences)
        crfcw01 = content_word_overlap.avg_adjacent_content_word_overlap(sentences)

        all_scores = [lsass1, lsass1d, lsa_giv, lsa_giv_d, synstruta, crfcw01]
        return all_scores

    def compute_final_score(self, sentences: list[Sentence]) -> float:
        """
        Compute a final combined score.
        :type sentences: the sentences to score.
        :return: the final score.
        """
        all_scores = self.compute_scores(sentences)

        return sum(v * w for v, w in
                   zip(all_scores, self.weights.values())) / sum(self.weights.values())


class LSAScorer:
    """
    Computes a score using LSAGN and LSASS1.
    """

    def __init__(self, vectorizer: SBERTVectorizer):
        self.adjacent_sentences = LSAAdjacentSentences(vectorizer)

    def score(self, sentences: list[Sentence]) -> float:
        lsass1 = self.adjacent_sentences.lsa_adjacent(sentences)[0]
        lsagn = taaco_givenness.giv_avg(sentences)
        return lsagn + lsass1


def test_TextScorer():
    vectorizer = CachedSBERTVectorizer()
    sentences = [
        "Baljväxter är den grupp inom grönsaker som skiljer sig mest från de andra.",
        "Baljväxter är ärtor, bönor och linser.",
        "Gemensamt för dessa är att de växer i en så kallad balja, en kapsel som man sedan öppnar för att ta ut de mogna fröna för att äta."
    ]
    sentences = " ".join(sentences)
    parser = Parser()
    doc = parser.parse(sentences)
    scorer = TextScorer(vectorizer)
    final = scorer.compute_final_score(doc.sentences)
    print(f'TextScorer, final score: {round(final, 4)}')


if __name__ == "__main__":
    test_TextScorer()

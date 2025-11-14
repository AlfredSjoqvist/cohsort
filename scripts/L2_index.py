import content_word_overlap
import syntactic_similarity
# import word_frequencies # placeholder
from cached_SBERT import CachedSBERTVectorizer
import math
from stanza.models.common.doc import Sentence
from stanza.models.common.doc import Word


def compute_l2(sentences: list[Sentence]):
    """
    Compute the L2-score.
    :param sentences: a list of sentences.
    """
    similarity = syntactic_similarity.avg_syntax_similarity(sentences)  # synt.synt_struc_adj(sentences)
    word_overlap = content_word_overlap.avg_adjacent_content_word_overlap(sentences)  # content_word_overlap.avg_adjacent_content_word_overlap(sentences)
    frequencies = 0  # word_frequencies.word_freq(sentences) #placholder

    # Coh-Metrix L2 Reading Index = – 45.032 + (52.230 x Content Word Overlap Value) + (61.306 x Sentence Syntax Similarity Value) + (22.205 x CELEX Frequency Value)
    l2_value = -45.032 + (52.230 * word_overlap) + (61.305 * similarity) # + (22.205 * frequencies)
    l2_value = normalize(l2_value)
    return l2_value


def normalize(value: float) -> float:
    """
    Convert the L2_score into a score between 0 and 1.
    """
    # Needs an update.

    # We need to estimate max_values for each metric.
    # min = -45.032
    # usual max_values
    # SYNSSTRUTa = 1 (52.230)
    # CRFCW01 = 1 (61.305)
    # WRDFRQmc = 2 (44.410) ?? Unsure about this one
    # -45.032 + 52.230 + 61.305 + 44.410 = 112.913
    normalized_index = (value + 45.032) / (45.032 + 112.913)
    return normalized_index


def test_compute_l2():
    sentences = [
        "Baljväxter är den grupp inom grönsaker som skiljer sig mest från de andra.",
        "Baljväxter är ärtor, bönor och linser.",
        "Gemensamt för dessa är att de växer i en så kallad balja, en kapsel som man sedan öppnar för att ta ut de mogna fröna för att äta."
    ]
    print(compute_l2(sentences))


if __name__ == "__main__":
    test_compute_l2()

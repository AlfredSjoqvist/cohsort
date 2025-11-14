"""
Coh-metrix index 34 (CRFCWO1): Content word overlap of adjacent sentences.
"""

from stanza.models.common.doc import Sentence
from stanza.models.common.doc import Word
from parsing import Parser

# UPOS for open/content word classes. See https://universaldependencies.org/u/pos/
CONTENT_WORD_CLASSES = [
    'ADJ',
    'ADV',
    'INTJ',
    'NOUN',
    'PROPN',
    'VERB'
]


def avg_adjacent_content_word_overlap(sentences: list[Sentence]) -> float:
    """
    Compute the average proportion of overlapping content words for each
    sentence pair.

    :param sentences: a list of sentences.
    :return: the average proportion of overlapping content words. A float between 0 and 1.
    """

    # Return 0 if no pairs can be formed.
    if len(sentences) < 2:
        return 0

    # Compute the average overlap for each sentence pair.
    total_overlap = 0
    for i in range(0, len(sentences) - 1):
        sentence = sentences[i]
        next_sentence = sentences[i + 1]
        total_overlap += content_word_overlap(sentence, next_sentence)

    return total_overlap / (len(sentences) - 1)


def content_word_overlap(sentence_1: Sentence, sentence_2: Sentence) -> float:
    """
    Compute the proportion of overlapping content words in the pair of sentences.

    :param sentence_1: the first sentence.
    :param sentence_2: the second sentence.
    :return: a float between 0 and 1, which is proportion of overlapping content
            words in the pair of sentences.
    """

    # #OverlappingWords / (#WordsInSentence1 + #WordsInSentence2)

    overlaps = 0
    for word in sentence_1.words:
        word = word  # type: Word

        # If it's a content word, count its occurrences.
        if word.upos in CONTENT_WORD_CLASSES:

            # Count the occurrences in the next sentence.
            occurrences = 0
            for other_word in sentence_2.words:
                other_word = other_word  # type: Word

                if other_word.lemma == word.lemma and other_word.upos == word.upos:
                    occurrences += 1

            # Count the overlaps (including itself).
            if occurrences > 0:
                overlaps += occurrences + 1

    return overlaps / (len(sentence_1.words) + len(sentence_2.words))


# Testing -------------------------------------------------------------

def test_content_word_overlap():
    sentences = [
        "Baljväxter är den grupp inom grönsaker som skiljer sig mest från de andra.",
        "Baljväxter är ärtor, bönor och linser.",
        "Gemensamt för dessa är att de växer i en så kallad balja, en kapsel som man sedan öppnar för att ta ut de mogna fröna för att äta."
    ]
    parser = Parser()
    doc = parser.parse(" ".join(sentences))
    print("Content word overlap: ", content_word_overlap(doc.sentences[0], doc.sentences[1]))


def test_avg_adjacent_content_word_overlap():
    sentences = [
        "Baljväxter är den grupp inom grönsaker som skiljer sig mest från de andra.",
        "Baljväxter är ärtor, bönor och linser.",
        "Gemensamt för dessa är att de växer i en så kallad balja, en kapsel som man sedan öppnar för att ta ut de mogna fröna för att äta."
    ]
    parser = Parser()
    doc = parser.parse(" ".join(sentences))
    print("Avg adjacent word overlap: ", avg_adjacent_content_word_overlap(doc.sentences))


if __name__ == '__main__':
    # test_content_word_overlap()
    test_avg_adjacent_content_word_overlap()

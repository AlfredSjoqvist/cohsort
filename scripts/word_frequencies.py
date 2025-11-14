from collections import defaultdict

from stanza import Document
from stanza.models.common.doc import Sentence, Word
import math

import parsing

# UPOS for open/content word classes. See https://universaldependencies.org/u/pos/
CONTENT_WORD_CLASSES = [
    'ADJ',
    'ADV',
    'INTJ',
    'NOUN',
    'PROPN',
    'VERB'
]

# SUC POS tags mapped to universal POS tags.
SUC_POS = {
    "AB": "ADV",
    "DT": "DET",
    "HA": "ADJ",
    "HD": "ADJ",
    "HP": "PRON",
    "HS": "ADJ",
    "IE": "PART",
    "IN": "INTJ",
    "JJ": "ADJ",
    "KN": "CCONJ",
    "NN": "NOUN",
    "PC": "ADJ",
    "PL": "PART",
    "PM": "PROPN",
    "PN": "PRON",
    "PP": "ADP",
    "PS": "ADJ",
    "RG": "NUM",
    "RO": "NUM",
    "SN": "SCONJ",
    "UO": "X",
    "VB": "VERB"
}


# These are absolute frequencies.

class WordFrequencies:
    """
    Computes Coh-Metrix word frequency measures. The indexes available are:
    - WRDFRQa (index 95).
    - WRDFRQmc (index 96).

    The original Coh-Metrix uses the CELEX database. This implementation, however,
    uses the NyLLex token frequency table.
    The file for the frequency table is 'nyllex_v2.csv'.
    """

    def __init__(self, filename: str):
        """
        Initialize the frequency table by loading it from a CSV file.
        :param filename: the CSV file.
        """
        self.frequency_table = defaultdict(int)  # type: defaultdict[tuple[str, str], int]

        with open(filename) as source:
            # Skip first header line.
            source.readline()

            # Read each line in file, and map lemma to count.
            for line in source:
                columns = line.strip().split(',')

                # Only count words with known tags.
                if columns[1] in SUC_POS:
                    key = (columns[0], SUC_POS[columns[1]])  # (lemma, pos)
                    self.frequency_table[key] += int(columns[15])

    def frequency(self, word: Word) -> int:
        """
        The absolute frequency (word count) of the given word.
        :param word: the word.
        :return: the absolute frequency of the word.
        """
        return self.frequency_table[(word.lemma, word.upos)]

    def avg_log_word_frequency(self, sentences: list[Sentence]) -> float:
        """
        WRDFRQa (index 95). The average word log frequency for all words.

        :param sentences: a list of sentences.
        :return: the average word log frequency for words as a float.
        """

        total_frequency = 0
        word_count = 0
        for sentence in sentences:
            for word in sentence.words:

                frequency = self.frequency(word)

                # Ignore frequencies lower than 1 (technically this would only be 0).
                if frequency >= 1:
                    total_frequency += math.log(frequency)
                    word_count += 1

        return total_frequency / word_count

    def avg_log_min_word_frequency(self, sentences: list[Sentence]) -> float:
        """
        WRDFRQmc (index 96). The average minimum content word log frequency
        across sentences.

        :param sentences: a list of sentences.
        :return: the average minimum content word log frequency across sentences.
        """

        # Make sure there are sentences in the list.
        if len(sentences) == 0:
            raise ValueError("No sentences.")

        # Sum the min content word frequency for each sentence.
        total_frequency = 0
        for sentence in sentences:
            min_frequency = self.least_frequent_content_word(sentence)

            # Ignore frequencies lower than 1 (technically this would only be 0).
            if min_frequency >= 1:
                total_frequency += math.log(min_frequency)

        # Compute the average over sentences.
        return total_frequency / len(sentences)

    def least_frequent_content_word(self, sentence: Sentence) -> int:
        """
        Find the frequency of the least frequent word in the sentence.
        :param sentence: the sentence.
        :return: the frequency of the least frequent word.
        """

        # Make sure there are words in the sentences.
        if len(sentence.words) == 0:
            raise ValueError("empty sentence: ", sentence)

        # Find the least frequent content word.
        min_freq = float('inf')
        for word in sentence.words:

            # Only count if content word.
            if word.upos in CONTENT_WORD_CLASSES:
                frequency = self.frequency(word)

                # Only count frequency if it occurs in the frequency table.
                if frequency >= 1:
                    min_freq = min(min_freq, frequency)

        return min_freq


def create_test_setup() -> tuple[Document, WordFrequencies]:
    sentences = " ".join([
        "Biologi är vetenskapen om livet på jorden.",
        "Den studerar allt från enkla celler till hela ekosystem.",
        "Biologi handlar om att förstå hur organismer fungerar och interagerar med sin omgivning.",
        "Det innefattar studiet av gener och hur de påverkar egenskaper och beteenden.",
        "Biologi inkluderar också studiet av miljön och dess påverkan på levande organismer.",
        "Forskning inom biologi kan lösa problem som sjukdomar, klimatförändringar och förlust av biologisk mångfald.",
        "Biologi är också en tillämpad vetenskap, inklusive områden som bioteknik och medicin.",
        "Biologi omfattar flera discipliner, inklusive zoologi, botanik, ekologi, molekylärbiologi och biokemi.",
        "Biologer använder hypotesprövning för att testa sina teorier och göra nya upptäckter.",
        "Biologi är en dynamisk vetenskap som ständigt utvecklas genom nya upptäckter och teknologiska framsteg.",
        "Det är en viktig disciplin för att förstå vår plats i världen och ta itu med globala utmaningar.",
        "Studier inom biologi har lett till stora framsteg inom medicin, livsmedelsproduktion och miljövård.",
        "Biologer studerar inte bara levande organismer, utan också hur de interagerar med varandra och sin omgivning.",
        "De kan också använda tekniker som genteknik och mikroskopi för att studera biologiska system.",
        "Evolution är en grundläggande princip inom biologi och förklarar hur organismer förändras över tid.",
        "Forskningsområden inom biologi inkluderar neurovetenskap, beteendegenetik och ekotoxikologi.",
        "Biologi är också viktigt för att förstå effekterna av mänsklig aktivitet på miljön och ekosystemen.",
        "Många biologiska principer är tillämpliga på andra områden, inklusive teknik och design.",
        "Studier inom biologi kan också bidra till att lösa samhällsproblem, inklusive hunger och sjukdomar."
    ])
    parser = parsing.Parser(constituencies=False)
    doc = parser.parse(sentences)
    word_frequencies = WordFrequencies('nyllex_v2.csv')
    return doc, word_frequencies


def test_avg_word_frequency():
    doc, word_frequencies = create_test_setup()
    print("avg word freqs: ", word_frequencies.avg_log_word_frequency(doc.sentences))


def test_frequencies():
    doc, word_frequencies = create_test_setup()

    # print("och: ", word_frequencies.frequency('och'))
    # print("inte: ", word_frequencies.frequency('inte'))

    print("avg word freqs: ", word_frequencies.avg_log_word_frequency(doc.sentences))
    print("avg log min frequency: ", word_frequencies.avg_log_min_word_frequency(doc.sentences))


if __name__ == '__main__':
    # test_frequency()
    # test_avg_word_frequency()
    test_frequencies()

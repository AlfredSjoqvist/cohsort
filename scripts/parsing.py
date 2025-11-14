import benepar
import stanza
import pickle

from benepar import InputSentence
from nltk import Tree
from stanza import Document, DownloadMethod
from stanza.models.common.doc import Sentence

import taaco_givenness
import os

# Add constituency property to stanza Sentence.
Sentence.add_property('ben_constituency', default=None,
                      getter=lambda self: self._ben_constituency,
                      setter=(lambda self, value: ben_constituency_setter(self, value)))


def ben_constituency_setter(sentence: Sentence, value):
    sentence._ben_constituency = value


def load_summary(file: str) -> str:
    """
    Load a summary from a text-file.
    :param file: a filename as a string.
    :return: a summary formatted as one string.
    """
    with open(file, 'r', encoding="utf-8") as f:
        summary = f.read()
    return summary


def to_dict(stanza_tree: Document) -> list[list[dict]]:
    """
    Turn a stanza tree into the standardized-format.
    :param stanza_tree: a stanza Document class.
    :return: a python-standardized version of the sentences and their contents.
    """
    return stanza_tree.to_dict()


def save_tree(standardized_tree: Document):
    """
    Save a standardized tree as a pickle file.
    :param save_tree: a stanza Document class.
    """
    with open(os.path.join("standardized_test.pickle"), "wb") as handle:
        pickle.dump(standardized_tree, handle, protocol=pickle.HIGHEST_PROTOCOL)


def nice_print(segmented_summary: Document):
    """ 
    Print the segmented sentences in a readable manner, requires a stanza tree. 
    :param segmented_summary: a stanza Document class.
    """
    for i, sentence in enumerate(segmented_summary.sentences):
        print(f'====== Sentence {i + 1} tokens =======')
        print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')


def load_tree() -> Document:
    """
    Load a standardized tree from a pickle file.
    :return: a stanza Document class.
    """
    with open(os.path.join("standardized_test.pickle"), "rb") as handle:
        standardized_tree = pickle.load(handle)
        return standardized_tree


def to_text(stanza_tree: Document) -> list[str]:
    """
    Take a stanza document and turn it into a list of sentences.
    :param stanza_tree: a stanza Document class.
    :return: a list of sentences as strings.
    """
    text_sentences = []
    for sentence in stanza_tree.sentences:
        text_sentences.append(sentence.text)
    return text_sentences


class Parser:
    """
    Parses written texts into analyzable data. The parsing consists of:
    - Sentence and segmentation.
    - Tokenization.
    - POS-tagging.
    - Dependency parsing.
    - Constituency parsing.
    """

    def __init__(self, constituencies=True):
        # Runs stanza, which tokenize text into sentences and words, and annotates POS and Lemma.
        self.pipeline = stanza.Pipeline(lang='sv', processors='tokenize, pos, lemma, depparse',
                                        download_method=DownloadMethod.REUSE_RESOURCES)

        # Load the Berkeley Neural Parser for constituency parsing.
        if constituencies:
            print("Loading benepar...")
            self.benepar_parser = benepar.Parser('benepar_sv2')
            print("benepar loaded")
        else :
            self.benepar_parser = None

    def parse(self, text: str) -> Document:
        """
        Parses the text into a stanza Document.

        :param text: the text as one single string.
        :return: a stanza Document class.
        """

        doc = self.pipeline(text)
        if self.benepar_parser is not None:
            self.parse_constituencies(doc)
        return doc

    def parse_constituencies(self, doc: Document):
        """
        Parse the constituencies for each sentence in the document.
        :param doc: the document containing the sentences.
        """

        for sentence in doc.sentences:
            words = [word.text for word in sentence.words]
            pos_tags = [word.upos for word in sentence.words]

            input_sentence = InputSentence(words=words, tags=pos_tags)
            tree = self.benepar_parser.parse(input_sentence)
            sentence.ben_constituency = tree


## For testing ##

def test_parse_constituencies():
    sentences = "".join([
        "Baljväxter är den grupp inom grönsaker som skiljer sig mest från de andra.",
        "Baljväxter är ärtor, bönor och linser.",
        "Gemensamt för dessa är att de växer i en så kallad balja, en kapsel som man sedan öppnar för att ta ut de mogna fröna för att äta."
    ])

    parser = Parser()
    doc = parser.parse(sentences)

    for sentence in doc.sentences:
        tree = sentence.ben_constituency  # type: Tree
        print(tree)


def test_1():
    """ Create and save a standardized tree from scratch. """
    seg = Parser()

    summary = load_summary("baljväxter.txt")

    segmented_summary = seg.parse(summary)

    nice_print(segmented_summary)

    standardized_tree = to_dict(segmented_summary)

    save_tree(standardized_tree)

    print(f'Standardized tree\n{standardized_tree}')

    text_sentences = to_text(segmented_summary)

    print(f'text version of sentences:\n{text_sentences}')


def test_2():
    """ Load a standardized tree from pickle-file. """
    seg = Parser()
    standardized_tree = load_tree()
    print(standardized_tree)


def test_3():
    """Test the givenness metric."""
    seg = Parser()
    summary = load_summary("baljväxter.txt")

    standardized_tree = seg.parse(summary).sentences

    avg_word_giv = taaco_givenness.giv_avg_entire_text(standardized_tree)

    avg_sentence_giv = taaco_givenness.giv_avg(standardized_tree)

    stdev_sentence_giv = taaco_givenness.giv_stdev(standardized_tree)

    print(avg_word_giv)
    print(avg_sentence_giv)
    print(stdev_sentence_giv)


if __name__ == "__main__":
    test_parse_constituencies()

import numpy as np
import timeit

from stanza.models.common.doc import Sentence

import SBERT
from parsing import Parser


class CachedSBERTVectorizer(SBERT.SBERTVectorizer):
    """
    An extension of the SBERTVectorizer that caches the computed embeddings.
    SBERT embeddings can be fairly costly to compute, so by storing
    them we minimize execution time at the cost of memory. For our
    application this is useful because we test the same few embeddings
    several times.
    """

    def __init__(self):
        super().__init__()
        self.cache = {}  # type: dict[str, np.ndarray[float]]

    def vectorize(self, sentence) -> np.ndarray[float]:
        # See super method for doc-string.

        # If stanza sentence, use the raw text from it.
        if type(sentence) == Sentence:
            sentence = sentence.text

        # Retrieve embedding from cache.
        embedding = self.cache.get(sentence)

        # If embedding is not in cache, create it and then cache it.
        if embedding is None:
            embedding = super().vectorize(sentence)
            self.cache[sentence] = embedding

        return embedding

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()


def test_cached():
    vectorizer = CachedSBERTVectorizer()
    sentence = 'Det här är en exempelmening'

    # Take time in milliseconds for running.
    print(timeit.timeit(lambda: vectorizer.vectorize(sentence), number=1) * 1000)
    print(timeit.timeit(lambda: vectorizer.vectorize(sentence), number=10) * 1000)


def test_cached_stanza():
    vectorizer = CachedSBERTVectorizer()
    sentences = [
        "Baljväxter är den grupp inom grönsaker som skiljer sig mest från de andra.",
        "Baljväxter är ärtor, bönor och linser.",
        "Gemensamt för dessa är att de växer i en så kallad balja, en kapsel som man sedan öppnar för att ta ut de mogna fröna för att äta."
    ]
    parser = Parser()
    doc = parser.parse(" ".join(sentences))

    print(vectorizer.vectorize(doc.sentences[0])[:5])
    print(vectorizer.vectorize(doc.sentences[1])[:5])
    print(vectorizer.vectorize(doc.sentences[2])[:5])


if __name__ == '__main__':
    test_cached_stanza()

import numpy as np
from sentence_transformers import SentenceTransformer
from stanza.models.common.doc import Sentence


class SBERTVectorizer:
    """
    Sentence-BERT for creating sentence embeddings.

    Example:
    sentence = "Det här är en exempelmening"
    vectorizer = SBERTVectorizer()
    embedding = vectorizer.vectorize(sentence)
    print(embedding)
    #out: [0.1339728832244873, -0.07478315383195877, ..., ]
    """

    def __init__(self):
        self.model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

    def vectorize(self, sentence) -> np.ndarray[float]:
        """
        Create a sentence embedding for the given sentence.

        :param sentence: either a string for the sentence, or a stanza.Sentence.
        :return: an np.ndarray of floats which is the vector.
        """

        # If stanza sentence, use the raw text from it.
        if type(sentence) == Sentence:
            sentence = sentence.text  #" ".join([word.lemma for word in sentence.words])

        return self.model.encode(sentence, convert_to_numpy=True)

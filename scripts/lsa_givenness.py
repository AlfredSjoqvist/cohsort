from stanza.models.common.doc import Sentence
import numpy as np

from SBERT import SBERTVectorizer
from cached_SBERT import CachedSBERTVectorizer
from parsing import Parser


def project(vector_1: np.ndarray, vector_2: np.ndarray) -> np.ndarray:
    """
    Project the first vector onto the second.

    :param vector_1: the vector to be projected.
    :param vector_2: the vector to be projected onto.
    :return: the resulting projected vector.
    """

    # proj = (v1·v2 / (|v2|^2)) * v2
    dot = np.dot(vector_1, vector_2)
    len2 = np.dot(vector_2, vector_2)
    return vector_2.copy() * (dot / len2)


def orthogonalize(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """
    Orthogonalize the subspace specified by the vectors. This uses the Gram-Schmidt
    orthogonalization process. Note that the resulting orthogonalized vectors are not
    normalized.

    :param vectors: a list of vectors specifying the subspace.
    :return: a new list of orthogonal vectors specifying the subspace.
    """

    remaining = vectors.copy()
    ortho_space = [remaining.pop().copy()]

    while len(remaining) > 0:
        next_vec = remaining.pop().copy()  # Copy so we don't modify the original embedding.
        ortho_projections = [project(next_vec, ortho_vec) for ortho_vec in ortho_space]

        # Subtract projections from the vector.
        for projection in ortho_projections:
            next_vec -= projection

        ortho_space.append(next_vec)

    return ortho_space


def project_onto_subspace(vector: np.ndarray, other_vectors: list[np.ndarray]) -> np.ndarray:
    """
    Project a vector onto the subspace defined by a list of vectors.
    The subspace vectors do not have to be orthogonal.

    :param vector: the vector to project.
    :param other_vectors: a list of vectors specifying the subspace.
    :return: the projected vector onto the subspace.
    """

    orthogonal_subspace = orthogonalize(other_vectors)
    sub_space_projection = np.zeros_like(vector)

    # Project onto subspace.
    for ortho_vec in orthogonal_subspace:
        sub_space_projection += project(vector, ortho_vec)

    return sub_space_projection


class LSAGivenness:

    def __init__(self, vectorizer: SBERTVectorizer):
        self.vectorizer = vectorizer

    def givenness(self, sentences: list[Sentence]) -> tuple[float, float]:
        """
        Compute the average and standard deviation givenness of the text.

        :param sentences: the sentences in the text. There must be at least two sentences.
        :return: a tuple of the average and standard deviation: (avg_givenness, std_givenness)
        """

        if len(sentences) < 2:
            raise ValueError("There must be at least two sentences: ", len(sentences))

        # Compute the givenness for each sentence in relation to previous sentences.
        givenness = []
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            previous_sentences = sentences[:i]
            givenness.append(self.compute_givenness(sentence, previous_sentences))

        # Compute mean and standard deviations.
        avg = np.average(givenness)
        std = np.std(givenness)
        return avg, std

    def compute_givenness(self, sentence, previous_sentences: list[Sentence]) -> float:
        """
        Compute the givenness of the sentence in relation to the previous sentences.

        :param sentence: the current sentence.
        :param previous_sentences: the previous sentences.
        :return: the givenness of the sentence. A value between [0, 1].
        """
        embedding = self.vectorizer.vectorize(sentence)
        other_embeddings = [self.vectorizer.vectorize(sent) for sent in previous_sentences]

        # Project onto other embeddings.
        projection = project_onto_subspace(embedding, other_embeddings)

        # Compute the amount of new information.
        new_information = embedding - projection
        len_old = np.linalg.norm(projection)  # norm computes the length of the vector.
        len_new = np.linalg.norm(new_information)

        # Return the ratio between old and new information.
        return len_old / (len_new + len_old)


def test_project():
    print("test_projection")
    vector1 = np.array([1, 0, 0], dtype=float)
    vector2 = np.array([0, 1, 0], dtype=float)
    vector3 = np.array([0, 0, 1], dtype=float)

    vector4 = np.array([1, 1, 1], dtype=float)

    print(project(vector4, vector1), ' should be ', vector1)
    print(project(vector1, vector2), ' should be [0. 0. 0.]')
    print(project(-vector4, vector2), ' should be ', -vector2)


def test_orthogonalize():
    print("test_orthogonalize")
    subspace = [
        np.array([1, 0, 1], dtype=float),
        np.array([2, 1, 2], dtype=float),
        np.array([-1, 0, 0], dtype=float)
    ]

    ortho_subspace = orthogonalize(subspace)
    print(ortho_subspace)


def test_project_onto_subspace():
    print('test_project_onto_subspace')
    subspace = [
        np.array([1, 0, 1, 1], dtype=float),
        np.array([2, 0, 2, 2], dtype=float),
        np.array([2, 0, 2, -1], dtype=float),
        np.array([-1, 0, 0, 2], dtype=float),
    ]

    vector_1 = np.array([1, 0, 1, 1], dtype=float)
    vector_2 = np.array([0, 1, 0, 0], dtype=float)

    projection_1 = project_onto_subspace(vector_1, subspace)
    projection_2 = project_onto_subspace(vector_2, subspace)

    print(projection_1)
    print(projection_2)

    # Control dot products.
    print('Checking dot product for vector1')
    for sub_vec in subspace:
        print(round(np.dot(projection_1, sub_vec)), 'should not be 0')

    print('Checking dot product for vector2')
    for sub_vec in subspace:
        print(round(np.dot(projection_2, sub_vec)), 'should be 0')


def create_test_setup():
    sentences = ' '.join([
        "Baljväxter är den grupp inom grönsaker som skiljer sig mest från de andra.",
        "Baljväxter är ärtor, bönor och linser.",
        "Gemensamt för dessa är att de växer i en så kallad balja, en kapsel som man sedan öppnar för att ta ut de mogna fröna för att äta."
    ])
    doc = Parser(constituencies=False).parse(sentences)
    vectorizer = CachedSBERTVectorizer()
    givenness = LSAGivenness(vectorizer)
    return givenness, doc


def test_compute_givenness():
    print('test_compute_givenness')
    givenness, doc = create_test_setup()
    sentence_1 = doc.sentences[0]
    sentence_2 = doc.sentences[1]

    print('sentence 1:', sentence_1.text)
    print('sentence 2:', sentence_2.text)

    print('given(1, 2): ', givenness.compute_givenness(sentence_1, [sentence_2]))
    print('given(1, 1): ', givenness.compute_givenness(sentence_1, [sentence_1]))
    print('given(2, 2): ', givenness.compute_givenness(sentence_2, [sentence_2]))


def test_avg_givenness():
    print('test_avg_givenness')
    givenness, doc = create_test_setup()
    print(givenness.givenness(doc.sentences))


if __name__ == '__main__':
    # test_project()
    # test_orthogonalize()
    # test_project_onto_subspace()
    # test_compute_givenness()
    test_avg_givenness()

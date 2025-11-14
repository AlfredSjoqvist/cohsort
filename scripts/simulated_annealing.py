from typing import Callable
import py_search.optimization
from py_search.base import Problem, Node
from SBERT import SBERTVectorizer
import random
import timeit
from text_scorer import TextScorer
from lsa_adjacent_sentences import LSAAdjacentSentences
from parsing import Parser
from cached_SBERT import CachedSBERTVectorizer
from stanza.models.common.doc import Sentence


class OrderingProblem(Problem):
    """
    Specifies the search-problem for search_py.
    """

    def __init__(self, initial_order: list[Sentence], scorer: Callable[[list[Sentence]], float]):
        """
        :param initial_order: the initial order of sentences.
        :param scorer: the scoring function for orderings of sentences.
        """
        super().__init__(initial_order)
        self.scorer = scorer

    def random_successor(self, node: Node):
        current_order = node.state  # type: list[Sentence]
        new_order = list(current_order)

        # Choose random indexes for swapping.
        random_index_1 = random.randint(0, len(new_order) - 1)
        random_index_2 = random.randint(0, len(new_order) - 1)

        # Swap places for sentences.
        new_order[random_index_1] = current_order[random_index_2]
        new_order[random_index_2] = current_order[random_index_1]

        return Node(new_order)

    def node_value(self, node: Node):
        return -self.scorer(node.state)  # Use negative value because sim ann is gradient descent.

    def goal_test(self, state_node, goal_node=None):
        return False  # There is no way of knowing beforehand if a certain order is correct.


class SimulatedAnnealing:
    """
    Finds a near optimal ordering of sentences using simulated annealing.

    It is preferable to use a CachedSBERTVectorizer for creating
    embeddings, because this algorithm will compute the same embeddings
    several times, making caching suitable.
    """

    def __init__(self, scoring_function: Callable[[list[Sentence]], float]):
        """
        :param scoring_function: the scoring function used for computing scores for sentence orderings.
        """
        self.scoring_function = scoring_function

    def find_good_order(self, sentences: list[Sentence]) -> list[Sentence]:
        """
        Find a close to optimal order of the sentences.
        :param sentences: a list sentences to be sorted.
        :return: a new list of sentences in near-optimal order.
        """
        problem = OrderingProblem(sentences, self.scoring_function)
        search = py_search.optimization.simulated_annealing(problem, temp_length=len(sentences) ** 2)
        results = next(search)
        return results.state_node.state


def test_sim_ann(vectorizer):
    sentences = [
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
    ]

    sentences = " ".join(sentences)

    scorer = TextScorer(vectorizer)
    search = SimulatedAnnealing(scorer.compute_final_score)
    parser = Parser()
    doc = parser.parse(sentences)
    sentences = doc.sentences

    new_order = search.find_good_order(sentences)
    print("Old order: ", [sentence.text for sentence in sentences])
    print("New order: ", [sentence.text for sentence in new_order])
    print("Old score: ", search.scoring_function(sentences))
    print("New score: ", search.scoring_function(new_order))


def test_timed():
    cached_vect = CachedSBERTVectorizer()
    standard_vect = SBERTVectorizer()

    print("standard: ", timeit.timeit(lambda: test_sim_ann(standard_vect), number=1))
    print("cached: ", timeit.timeit(lambda: test_sim_ann(cached_vect), number=1))

    # Results.
    # standard: 58.097374957986176 seconds.
    # cached: 0.41861462499946356 seconds
    # standard is 141 times slower.


if __name__ == '__main__':
    test_sim_ann(CachedSBERTVectorizer())
    # test_timed()

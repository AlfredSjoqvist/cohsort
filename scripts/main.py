from stanza.models.common.doc import Sentence

import pprint
from parsing import Parser
from simulated_annealing import SimulatedAnnealing
from cached_SBERT import CachedSBERTVectorizer
from text_scorer import TextScorer
import brute_force
from parsing import load_summary

class ElsaScrum:
    """
    The main ElsaScrum application.
    """

    def __init__(self):

        # For creating semantically meaningful sentence embeddings.
        self.vectorizer = CachedSBERTVectorizer()

        # The parser used for parsing summaries.
        self.parser = Parser()

        # The scorer used for scoring the ordering of sentences.
        self.scorer = TextScorer(self.vectorizer)

        # The search algorithm.
        self.search = SimulatedAnnealing(self.scorer.compute_final_score)

        # Automatically clear the vectorizer cache before each reordering.
        self.auto_clear_vect_cache = True

    def reorder(self, summary: str) -> str:
        """
        Reorder the sentences in the summary to improve cohesion. 
        Does not include the first and last sentence.
        :param summary: the summary consisting of a string.
        :return: the same summary but with reordered sentences.
        """
        doc = self.parser.parse(summary)
        #sentences = doc.sentences[1:-1]
        #improved_order = self.reorder_sentences(sentences)
        improved_order = self.reorder_sentences(doc.sentences)
        #improved_complete = " ".join([doc.sentences[0].text, improved_order_text, doc.sentences[-1].text])

        return " ".join([sentence.text for sentence in improved_order])
    
    def reorder_sentences(self, sentences: list[Sentence]) -> list[Sentence]:
        """
        Reorder the sentences in an already parsed list of sentences to improve cohesion.
        :param sentences: a list of parsed sentences.
        :return: a new list of reordered sentences.
        """
        # Clear the vectorizer cache.
        if self.auto_clear_vect_cache:
            self.vectorizer.clear_cache()

        # Find a new order.
        if len(sentences) < 8:
            return brute_force.brute_force_search(sentences, self.scorer)
        else:
            return self.search.find_good_order(sentences)


def real_shuffle():
    app = ElsaScrum()
    files = []
    for i in range(1,16):
        file_name = "summary" + str(i) + ".txt"
        files.append(file_name)
     

    for file in files:
        summary = load_summary(file)
        new_summary = app.reorder(summary)
        if len(file) == 12:
            write_file = file[0:8] + "_LSA.txt"
        else:
            write_file = file[0:9] + "_LSA.txt"        
        with open(write_file, 'w') as f:
            f.write(new_summary)

def test_elsascrum():
    summary = " ".join([
        "Biologi är vetenskapen om livet på jorden.",
        "Den studerar allt från enkla celler till hela ekosystem.",
        "Biologi handlar om att förstå hur organismer fungerar och interagerar med sin omgivning.",
        "Det innefattar studiet av gener och hur de påverkar egenskaper och beteenden.",
        "Biologi inkluderar också studiet av miljön och dess påverkan på levande organismer."
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

    app = ElsaScrum()
    new_summary = app.reorder(summary)
    pprint.pprint(new_summary)


if __name__ == '__main__':
    #test_elsascrum()
    real_shuffle()

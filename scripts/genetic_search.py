from SBERT import SBERTVectorizer
import random
import math
from text_scorer import TextScorer
from parsing import Parser
from cached_SBERT import CachedSBERTVectorizer
from stanza.models.common.doc import Sentence
import random


def fitness_function(individual : list[Sentence], vectorizer : CachedSBERTVectorizer) -> float:
    """
    Calculate the fitness score of an individual in the population.
        
    :param individual: A list of Sentence objects representing an individual.
    :param vectorizer: An instance of CachedSBERTVectorizer for vectorizing sentences.
    :return: A float representing the fitness score of the individual. A higher score is better.
    """
    scorer = TextScorer(vectorizer)
    return scorer.compute_final_score(individual)

# Genetic operators
def selection(population: list[list[Sentence]], fitnesses : list[float]) -> list[list[Sentence]]:
    """
    Perform selection on the population based on their fitness scores.
    
    This function uses the roulette wheel selection method to select individuals
    from the population based on their fitness scores.
    
    :param population: A list of lists, each containing Sentence objects representing a potential solution.
    :param fitnesses: A list of float values representing the fitness scores of each individual in the population.
    :return: A list of selected individuals from the population.
    """
    selected_indices = random.choices(range(len(population)), weights=fitnesses, k=len(population))
    return [population[i] for i in selected_indices]

def crossover(parent1: list[Sentence], parent2: list[Sentence]) -> tuple[list[Sentence], list[Sentence]]:
    """
    Perform one-point crossover between two parents to generate two children.
    
    This function generates two offspring by exchanging a segment of consecutive elements
    between two parent individuals.
    
    :param parent1: A list of Sentence objects representing the first parent.
    :param parent2: A list of Sentence objects representing the second parent.
    :return: A tuple containing two lists of Sentence objects representing the generated offspring.
    """
    size = len(parent1)

    child1, child2 = [None] * size, [None] * size

    start, end = sorted(random.sample(range(size), 2))

    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    # Fill the remaining slots of child1 and child2 with the values from parent2 and parent1 respectively
    for i in range(size):
        if parent2[i] not in child1:
            idx = end % size
            while child1[idx] is not None:
                idx = (idx + 1) % size
            child1[idx] = parent2[i]

        if parent1[i] not in child2:
            idx = end % size
            while child2[idx] is not None:
                idx = (idx + 1) % size
            child2[idx] = parent1[i]

    return child1, child2

def mutation(individual: list[Sentence], mutation_rate: float):
    """
    Apply mutation to an individual by swapping elements.
    
    This function mutates an individual by swapping elements in the list based on a given mutation rate.
    
    :param individual: A list of Sentence objects representing an individual.
    :param mutation_rate: A float value representing the probability of mutation.
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]

def main(sentences: list[Sentence]):
    """
    Main function for the genetic algorithm to optimize the ordering of sentences.
    
    This function initializes the population, applies genetic operators and finds the best solution
    for the given number of generations.
    
    :param sentences: A list of Sentence objects representing the input sentences.
    """
    # Parameters
    vectorizer = CachedSBERTVectorizer()
    num_generations = 200
    population_size = 20
    crossover_rate = 0.8
    mutation_rate = 0.2
    best_candidates_to_save = 10

    # Initialize population
    population = [random.sample(sentences, len(sentences)) for _ in range(population_size)]

    # To store the best individuals
    best_individuals = []

    for generation in range(num_generations):
        # Evaluate fitness
        fitnesses = [fitness_function(individual, vectorizer) for individual in population]

        # Save the ten best candidates of this generation
        sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        top_individuals = [individual for individual, _ in sorted_population[:best_candidates_to_save]]
        best_individuals.extend(top_individuals)
        best_individuals = [list(x) for x in set(tuple(x) for x in best_individuals)]

        # Select parents
        parents = selection(population, fitnesses)

        # Apply crossover
        offspring = []
        for i in range(0, population_size, 2):
            if random.random() < crossover_rate:
                offspring.extend(crossover(parents[i], parents[i + 1]))
            else:
                offspring.extend([parents[i], parents[i + 1]])

        # Apply mutation
        for individual in offspring:
            mutation(individual, mutation_rate)

        # Replace population
        population = offspring

    # Find the best solution
    best_individual = max(best_individuals, key=lambda x: fitness_function(x, vectorizer))
    best_fitness = fitness_function(best_individual, vectorizer)

    best_individual_text = []
    for sentence in best_individual:
        best_individual_text.append(sentence.text)

    print("Best individual:", best_individual_text)
    print("Fitness:", best_fitness)


if __name__ == "__main__":
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
    parser = Parser()
    doc = parser.parse(sentences)
    sentences = doc.sentences
    #print(sentences)
    # vectorizer = CachedSBERTVectorizer()
    main(sentences)

"""
Syntactic structure similarity adjacent: SYNSTRUTa (index 74)

This is the proportion of intersection tree nodes between all adjacent sentences.

The original Coh-Metrix index analyzes the constituency tree of the sentence. The
current implementation, however, can analyze both constituencies and dependency relations.

For constituencies to be analyzed, each sentence must have the ben_constituency property
assigned, which is done by parsing.Parser. Note that this is an NLTK constituency Tree,
as opposed to the stanza constituency ParseTree.

For dependencies, the sentences must have their 'deprel' property assigned, which
describes their dependency relations to their heads. Use the 'depparse' processor in the
Stanza pipeline to achieve this.
"""

import numpy as np
from networkx import DiGraph
import networkx as nx
from nltk import Tree
from stanza.models.common.doc import Sentence
from stanza.models.common.doc import Word


def avg_syntax_similarity(sentences: list[Sentence], structure_type: str = 'constituency') -> float:
    """
    Compute the average syntactic similarity between each adjacent sentence pair
    (SYNSTRUTa, index 74). This can be based either on constituencies or dependency relations.
    Use the structure_type parameter to dictate this. By default, constituencies are analyzed.

    :param sentences: a list of sentences.
    :param structure_type: the type of syntax structure to analyze. Can either be 'constituency' or 'dependency'.
    :return: a float of the average syntactic similarity.
    """

    # Compute the similarity for each adjacency pair.
    similarities = []
    for i in range(len(sentences) - 1):
        sentence = sentences[i]
        next_sentence = sentences[i + 1]
        similarities.append(syntax_similarity(sentence, next_sentence, structure_type))

    return np.average(similarities)


def syntax_similarity(sentence_1: Sentence, sentence_2: Sentence, structure_type: str = 'constituency') -> float:
    """
    Compute the syntactic similarity between the two sentences.

    This can be based either on constituencies or dependency relations. Use the
    structure_type parameter to dictate this.

    :param sentence_1: the first sentence.
    :param sentence_2: the second sentence.
    :param structure_type: the type of syntax structure to analyze. Can either be 'constituency' or 'dependency'.
    :return: float denoting the syntactic similarity between the sentences.
    """

    # Construct constituency trees.
    if structure_type == 'constituency':
        tree_1 = construct_constituency_tree(sentence_1)
        tree_2 = construct_constituency_tree(sentence_2)
    # Construct dependency trees.
    elif structure_type == 'dependency':
        tree_1 = construct_dependency_tree(sentence_1)
        tree_2 = construct_dependency_tree(sentence_2)
    else:
        raise ValueError(f'Unknown structure_type: {structure_type}')

    # Construct the common tree.
    common_tree = largest_common_subtree(tree_1, tree_2)

    # Compute the sizes of the trees.
    size_common = len(common_tree)
    size_1 = len(tree_1)
    size_2 = len(tree_2)

    # Similarity = #NodesInCommonTree / (#NodesInTree1 + #NodesInTree2 - #NodesInCommonTree)
    return size_common / (size_1 + size_2 - size_common)


def construct_constituency_tree(sentence: Sentence) -> DiGraph:
    """
    Construct a constituency tree (DiGraph) from the constituencies if the sentence.
    :param sentence: the sentence. It must have its ben_constituency property assigned.
    :return: the constructed tree as a DiGraph.
    """
    tree = DiGraph()
    constituencies = sentence.ben_constituency  # type: Tree
    root = (constituencies.label(), )
    create_constituency_children(root, constituencies, tree)
    return tree


def create_constituency_children(parent: tuple[str, ...], constituencies: Tree, tree: DiGraph):
    """
    Create nodes for the child constituencies of the parent node. This will in turn recursively create
    nodes for the children's child constituencies, and so on. The nodes are added and connected to
    passed in tree graph.

    :param parent: the parent node in the tree graph.
    :param constituencies: the nltk constituency tree.
    :param tree: the DiGraph tree which the nodes are added to.
    """
    for child in constituencies:
        if type(child) is Tree:
            label = child.label()
            node = tuple([label] + list(parent))
            tree.add_node(node)
            tree.add_edge(parent, node)

            # Recursively add grand-children.
            create_constituency_children(node, child, tree)


def construct_dependency_tree(sentence: Sentence) -> DiGraph:
    """
    Construct a dependency tree from the sentences dependency relations.

    :param sentence: the sentence to use. This must contain the 'deprel' property.
    :return: a DiGraph that represents dependency tree for the sentence.
    """
    tree = DiGraph()

    # Add words as nodes.
    for word in sentence.words:
        # tree.add_node(create_node(word))
        create_dependency_node(word, sentence, tree)

    # Connect nodes.
    for node in tree.nodes:
        if len(node) > 1:
            parent = node[1:]
            tree.add_edge(parent, node)

    return tree


def create_dependency_node(word: Word, sentence: Sentence, tree: DiGraph):
    """
    Create and add a dependency node for the word in the tree.

    A node is a tuple of strings, where each string is a dependency relation tag.
    The tuple represents all ancestor relations from the given word upwards to the root word.
    As an example, the node ('advmod', 'acl:relcl', 'obl', 'root'), where the current word has an
    'advmod' relation to its head, and that in turn has an 'acl:relcl' relation its
    own head, and so on.

    Note: this function does not connect the node to any other node.

    :param word: the word to create a node for.
    :param sentence: the sentence in which the word occurs.
    :param tree: the graph to add the node to.
    """
    node = [word.deprel]

    # Collect all ancestor dependency relations of the word.
    head = word.head
    while head != 0:
        word = sentence.words[head - 1]
        node.append(word.deprel)
        head = word.head

    # Add the node to tree.
    tree.add_node(tuple(node))

    # Todo: possible problem with duplicate node-to-root paths in graph.
    # for example, a sentence that contains several punctuations.


def largest_common_subtree(tree_1: DiGraph, tree_2: DiGraph) -> DiGraph:
    """
    Compute the largest common subtree of the two trees.

    :param tree_1: the first tree.
    :param tree_2: the second tree.
    :return: a DiGraph representing the largest common subtree.
    """

    # Code based on Florian Magin's stackoverflow answer:
    # https://stackoverflow.com/questions/43108481/maximum-common-subgraph-in-a-directed-graph
    matching_graph = nx.Graph()

    # Collect all common adjacent nodes for both trees.
    for node_1, node_2 in tree_2.edges():
        if tree_1.has_edge(node_1, node_2):
            matching_graph.add_edge(node_1, node_2)

    # Divide common graph into connected components.
    components = nx.connected_components(matching_graph)

    # Find the root component.
    root_component = None
    for component in components:
        if ('root', ) in component:
            root_component = component
            break

    # Create subgraph from the component.
    return nx.induced_subgraph(matching_graph, root_component)
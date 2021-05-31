import pathlib
from dataclasses import dataclass
from typing import Union, Optional, Any, BinaryIO

import networkx as nx
import numpy as np

from compy.representations.common import RepresentationBuilder, Graph


@dataclass
class Vocabulary:
    """
    Vocabulary for a set of sequence graphs.

    Sequence graphs can have different kinds of node kinds and edges.
    The vocabulary stores an array for each of those so they can be uniquely mapped to an integer (index with the array).
    """

    node_kinds: np.ndarray
    edge_kinds: np.ndarray

    @staticmethod
    def from_graph(graph: Graph) -> 'Vocabulary':
        return Vocabulary(node_kinds=np.array(graph.get_node_types()), edge_kinds=np.array(graph.get_edge_types()))

    @staticmethod
    def load(file: Union[str, BinaryIO, pathlib.Path]) -> 'Vocabulary':
        return Vocabulary(**np.load(file, allow_pickle=True))

    @staticmethod
    def unnamed(num_node_kinds: int, num_edge_kinds: int) -> 'Vocabulary':
        return Vocabulary(
            np.array(['node' + str(i) for i in range(num_node_kinds)]),
            np.array(['edge' + str(i) for i in range(num_edge_kinds)])
        )

    def save(self, file: Union[str, BinaryIO, pathlib.Path]):
        np.savez(file, node_kinds=self.node_kinds, edge_kinds=self.edge_kinds)


@dataclass
class SequenceGraph:
    """
    A sequence graph stores the elements of a sequence together with a graph built on top of those.

    An example is an AST graph, which is a sequence of tokens combined with a graph structure mapping those tokens
    to higher-level syntactical nodes.

    Attributes:

        :param nodes: Array with shape [N] of nodes, where the first ``seq_len`` form the underlying sequence
        :param edges: Array with shape [3, E] of edges, formatted as [edge_kind, source, dest]
        :param seq_len: Length of the underlying sequence
    """

    nodes: np.ndarray
    edges: np.ndarray
    seq_len: int

    __slots__ = ('nodes', 'edges', 'seq_len')

    def __init__(self, nodes: np.ndarray, edges: np.ndarray, seq_len: int):
        self.nodes = nodes
        self.edges = edges
        self.seq_len = seq_len

    def get_sequence_nodes(self):
        return self.nodes[:self.seq_len]

    def get_non_sequence_nodes(self):
        return self.nodes[self.seq_len:]

    def get_sequence_tokens(self, vocab: Vocabulary):
        return [vocab.node_kinds[i] for i in self.nodes[:self.seq_len]]

    def get_non_sequence_tokens(self, vocab: Vocabulary):
        return [vocab.node_kinds[i] for i in self.nodes[self.seq_len:]]

    def get_all_node_tokens(self, vocab):
        return [vocab.node_kinds[i] for i in self.nodes]

    def get_node_positions(self):
        """Return a list of the index of each node in the underlying sequence.

        The index is None for nodes not part of the sequence.
        """
        return [i if i < self.seq_len else None for i in range(self.nodes.shape[0])]

    def to_graph(self, vocab: Vocabulary) -> Graph:
        g = nx.MultiDiGraph()

        for idx, node in enumerate(self.nodes):
            if idx < self.seq_len:
                g.add_node(idx, attr=vocab.node_kinds[node] , seq_order=idx)
            else:
                g.add_node(idx, attr=vocab.node_kinds[node])

        for k, s, t in zip(*self.edges):
            g.add_edge(s, t, attr=vocab.edge_kinds[k])

        return Graph(g, list(vocab.node_kinds), list(vocab.edge_kinds))

    @staticmethod
    def from_graph(graph: Graph) -> 'SequenceGraph':
        node_to_int = { n: i for i, n in enumerate(graph.get_node_types()) }
        edge_to_int = { e: i for i, e in enumerate(graph.get_edge_types()) }

        nodes = []
        node_mapping = {}
        seq_len = 0
        for node, data in sorted(graph.G.nodes(data=True), key=lambda item: item[1].get('seq_order', float('inf'))):
            node_mapping[node] = len(nodes)
            nodes.append(node_to_int[data['attr']])

            if 'seq_order' in data:
                seq_len = len(nodes)

        edges = []
        for u, v, data in graph.G.edges(data=True):
            edges.append((edge_to_int[data['attr']], node_mapping[u], node_mapping[v]))

        return SequenceGraph(np.array(nodes, dtype=np.int32), np.array(list(zip(*edges)), dtype=np.int32), seq_len)


class SequenceGraphBuilder(RepresentationBuilder):
    __graph_builder: RepresentationBuilder
    __edge_kinds: tuple
    __node_kinds: tuple

    def __init__(self, graph_builder: RepresentationBuilder):
        super(SequenceGraphBuilder, self).__init__()
        self.__graph_builder = graph_builder
        self.__node_kinds = tuple()
        self.__edge_kinds = tuple()

    def num_tokens(self):
        return self.__graph_builder.num_tokens()

    def get_tokens(self):
        return self.__graph_builder.get_tokens()

    def print_tokens(self):
        self.__graph_builder.print_tokens()

    def vocabulary(self):
        return Vocabulary(np.array(self.__node_kinds, dtype=object), np.array(self.__edge_kinds))

    def string_to_info(self, src: Union[str, bytes], additional_include_dir: Optional[str] = None,
                       filename: Optional[str] = None) -> Any:
        return self.__graph_builder.string_to_info(src, additional_include_dir, filename)

    def info_to_representation(self, info, visitor):
        graph = self.__graph_builder.info_to_representation(info, visitor)

        new_node_kinds = tuple(graph.get_node_types())
        new_edge_kinds = tuple(graph.get_edge_types())

        assert new_node_kinds[:len(self.__node_kinds)] == self.__node_kinds, "vocabulary of new nodes must match"
        assert new_edge_kinds[:len(self.__edge_kinds)] == self.__edge_kinds, "vocabulary of new edges must match"

        self.__node_kinds = new_node_kinds
        self.__edge_kinds = new_edge_kinds

        return SequenceGraph.from_graph(graph)





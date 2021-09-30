import numpy as np
import numpy.testing as npt

from compy.representations.ast_graphs import ASTGraphBuilder, ASTDataCFGTokenVisitor
from compy.representations.sequence_graph import SequenceGraphBuilder, SequenceGraph

program_1fn_2 = """
int bar(int a) {
  if (a > 10)
    return a;
  return -1;
}
"""


def test_sequence_graph_builder():
    builder = SequenceGraphBuilder(ASTGraphBuilder())
    info = builder.string_to_info(program_1fn_2)
    rep = builder.info_to_representation(info, visitor=ASTDataCFGTokenVisitor)

    vocab = builder.vocabulary()

    assert rep.get_sequence_tokens(vocab) == [
        'int', 'identifier', 'l_paren', 'int', 'identifier', 'r_paren', 'l_brace',
        'if', 'l_paren', 'identifier', 'greater', 'numeric_constant', 'r_paren',
        'return', 'identifier', 'semi',
        'return', 'minus', 'numeric_constant', 'semi',
        'r_brace'
    ]

    builder.print_tokens()
    graph = rep.to_graph(vocab)
    str_nodes = graph.get_node_str_list()
    assert [str_nodes[i] for i in graph.get_leaf_node_list()] == rep.get_sequence_tokens(vocab)
    assert len(graph.get_edge_list()) == rep.edges.shape[-1]


def test_sequence_graph_to_undirected():
    test_edges = np.array([
        [0,0,1,2,0],
        [0,3,0,1,2],
        [1,1,2,2,3],
    ])
    test_nodes = np.arange(0, 4)
    graph = SequenceGraph(test_nodes.copy(), test_edges.copy(), 2)
    undirected = graph.to_undirected()

    assert undirected.seq_len == 2
    npt.assert_equal(undirected.nodes, np.arange(0, 4))
    npt.assert_equal(undirected.edges, np.array([
        [0, 0, 1, 2, 0, 3, 3, 4, 5, 3],
        [0, 3, 0, 1, 2, 1, 1, 2, 2, 3],
        [1, 1, 2, 2, 3, 0, 3, 0, 1, 2],
    ]))

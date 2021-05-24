from compy.representations.ast_graphs import ASTGraphBuilder, ASTDataCFGTokenVisitor
from compy.representations.sequence_graph import SequenceGraphBuilder

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
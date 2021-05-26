import numpy as np
import tensorflow as tf
import numpy.testing as npt
import networkx as nx

from compy.representations.common import Graph
from compy.representations.sequence_graph import SequenceGraph
from .tf2_sandwich_model import GGNNLayer, GlobalAttentionLayer, RNNLayer, Tf2SandwichModel, \
    ragged_graph_to_leaf_sequence, ragged_softmax, flatten_graph_batch

tf.compat.v1.enable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)


def assert_ragged_eq(a, b):
    npt.assert_equal([s.numpy() for s in a.nested_row_splits], [s.numpy() for s in b.nested_row_splits])
    npt.assert_equal(a.flat_values.numpy(), b.flat_values.numpy())


def test_flatten_graph_batch():
    n1, n2, n3, n4, n5, n6, n7, n8 = [(i,)*4 for i in range(8)]

    ragged_graph = tf.ragged.constant([
        [n1, n2, n3],
        [n4, n5],
        [n6, n7],
    ], inner_shape=(4,))
    ragged_graph_flat, ragged_graph_offsets, ragged_graph_repack = flatten_graph_batch(ragged_graph)

    assert_ragged_eq(ragged_graph, ragged_graph_repack(ragged_graph_flat))
    npt.assert_equal(ragged_graph_offsets.numpy(), [0, 3, 5])

    dense_graph = tf.constant([
        [n1, n2],
        [n3, n4]
    ])
    dense_flat, dense_offsets, dense_repack = flatten_graph_batch(dense_graph)
    npt.assert_equal(dense_graph.numpy(), dense_repack(dense_flat).numpy())
    npt.assert_equal(dense_offsets.numpy(), [0, 2])

    ragged_nested = tf.ragged.constant([
        [[n1, n2], [n3]],
        [[n4]]
    ], inner_shape=(4,))
    flat, offsets, repack = flatten_graph_batch(ragged_nested)
    assert_ragged_eq(repack(flat), ragged_nested)
    assert_ragged_eq(offsets, tf.ragged.constant([[0, 2], [3]]))

    ragged_mixed = tf.ragged.constant([
        [[n1, n2], [n3, n4]],
        [[n5, n6]]
    ], inner_shape=(2,4))
    flat, offsets, repack = flatten_graph_batch(ragged_mixed)
    assert_ragged_eq(repack(flat), ragged_mixed)
    assert_ragged_eq(offsets, tf.ragged.constant([[0, 2], [4]]))


def test_ggnn_layer():
    layer = GGNNLayer({
        'num_edge_types': 2,
        'time_steps': [3,1],
        'residuals': {1: [0]},
        'hidden_dim': 4,
        'add_type_bias': True,
        'dropout_rate': 0,
    })

    states = tf.random.uniform([10, 4], seed=0)
    edges = tf.constant([
        [0, 0, 1],
        [1, 0, 1],
        [0, 4, 5],
        [0, 6, 4],
        [0, 8, 9],
        [1, 2, 3]
    ], dtype=tf.int32)

    out = layer(states, edges, training=True)

    states_ragged = tf.RaggedTensor.from_row_lengths(states, [4, 4, 2])
    edges_ragged = tf.ragged.constant([
        [
            [0, 0, 1],
            [1, 0, 1],
            [1, 2, 3],
        ],
        [
            [0, 0, 1],
            [0, 2, 0],
        ],
        [
            [0, 0, 1],
        ]
    ], inner_shape=(3,))
    out_ragged = layer(states_ragged, edges_ragged)

    npt.assert_equal(out.numpy(), out_ragged.flat_values.numpy())


def test_global_attention_layer():
    layer = GlobalAttentionLayer()
    inputs = tf.random.uniform([10, 4], seed=0)
    graph_sizes = tf.constant([2, 4, 4])
    x = tf.RaggedTensor.from_row_lengths(inputs, graph_sizes)
    dense_x = x.to_tensor(0, shape=(3, 4, 4))
    dense_mask = tf.RaggedTensor.from_row_lengths([True] * 10, graph_sizes).to_tensor(False, shape=(3, 4))

    out = layer(x)
    dense_out = layer(dense_x, mask=dense_mask)

    out_gate0 = tf.squeeze(layer.gate_layer(inputs[:2]), -1)
    out_gate0 = tf.nn.softmax(out_gate0)
    out_v0 = layer.output_layer(inputs[:2])

    npt.assert_allclose(out[0].numpy(), (out_gate0[0] * out_v0[0] + out_gate0[1] * out_v0[1]).numpy(), rtol=1e-3)
    npt.assert_allclose(dense_out, out)


def test_rnn_layer():
    layer = RNNLayer({
        "hidden_dim": 4,
        "num_layers": 1,
        "dropout_rate": 0,
    })
    inputs = tf.random.uniform([10, 4], seed=0)
    graph_sizes = tf.constant([2, 4, 4])
    node_positions = tf.constant([1, 2, 0, 1, 2, 3, 3, 1, 0, 2])
    leaf_mask = tf.constant([True, True, False, True, True, True, True, True, False, True])

    x = tf.RaggedTensor.from_row_lengths(inputs, graph_sizes)
    out = layer(x, node_positions=tf.RaggedTensor.from_row_splits(node_positions, x.row_splits)).flat_values

    # non-leaf states should stay unchanged
    npt.assert_allclose(tf.boolean_mask(out, ~leaf_mask).numpy(), tf.boolean_mask(inputs, ~leaf_mask).numpy())

    x = tf.RaggedTensor.from_row_lengths(tf.random.uniform([1, 4], seed=0), [1])
    layer(x, node_positions=tf.ragged.constant([[1]]))


def test_ragged_graph_to_leaf_sequence():
    nodes = tf.range(10)
    graph_sizes = tf.constant([2,4,4])
    node_positions = tf.constant([1, 2, 0, 1, 2, 3, 3, 1, 0, 2])

    nodes = tf.RaggedTensor.from_row_lengths(nodes, graph_sizes)
    node_positions = tf.RaggedTensor.from_row_lengths(node_positions, graph_sizes)

    sequence = ragged_graph_to_leaf_sequence(nodes, node_positions)
    print(nodes[0, 0])
    npt.assert_equal(sequence.row_lengths().numpy(), np.array([2, 3, 3]))
    npt.assert_equal(sequence.flat_values.numpy(), np.array([
        0, 1,
        3, 4, 5,
        7, 9, 6,
    ]))


def f(x):
    return tf.reduce_sum(x, axis=-1)


def test_ragged_softmax():
    values = tf.random.uniform((8,), seed=213)
    sizes = tf.constant([4, 2, 2])
    indices = [(0, i) for i in range(4)] + [(1, i) for i in range(2)] + [(2, i) for i in range(2)]

    expected = tf.concat([
        tf.nn.softmax(values[0:4]),
        tf.nn.softmax(values[4:6]),
        tf.nn.softmax(values[6:8]),
    ], axis=0)

    dense_x = tf.scatter_nd(tf.constant(indices), values, shape=(3, 4))
    dense_mask = tf.scatter_nd(tf.constant(indices), tf.fill(values.shape, True), shape=dense_x.shape)
    dense_expected = tf.scatter_nd(tf.constant(indices), expected, shape=(3, 4))

    x = tf.RaggedTensor.from_row_lengths(values, sizes)
    npt.assert_allclose(ragged_softmax(x).flat_values.numpy(), expected.numpy(), rtol=1e-4)
    npt.assert_allclose(ragged_softmax(dense_x, mask=dense_mask).numpy(), dense_expected.numpy(), rtol=1e-4)

    # test that grad can be calculated
    with tf.GradientTape() as tape:
        tape.watch(x.flat_values)
        y = ragged_softmax(x)
    tape.gradient(y.flat_values, x.flat_values)


def test_train_model():
    dummy_graph = nx.MultiDiGraph()
    dummy_graph.add_node("n1", attr="a", seq_order=0)
    dummy_graph.add_node("n2", attr="b")
    dummy_graph.add_node("n3", attr="c")
    dummy_graph.add_edge("n1", "n2", attr="dummy")

    dummy_graph2 = nx.MultiDiGraph()
    dummy_graph2.add_node('n1', attr='a', seq_order=1)
    dummy_graph2.add_node('n2', attr='b', seq_order=2)
    dummy_graph2.add_edge('n1', 'n2', attr='dummy')
    dummy_graph2.add_edge('n2', 'n1', attr='dummy')

    data = [
        {
            "x": {
                "code_rep": SequenceGraph.from_graph(Graph(dummy_graph, ["a", "b", "c"], ["dummy"])),
                "aux_in": [],
            },
            "y": 0,
        },
        {
            "x": {
                "code_rep": SequenceGraph.from_graph(Graph(dummy_graph2, ["a", "b", "c"], ["dummy"])),
                "aux_in": [],
            },
            "y": 1,
        },
        {
            "x": {
                "code_rep": SequenceGraph.from_graph(Graph(dummy_graph2, ["a", "b", "c"], ["dummy"])),
                "aux_in": [],
            },
            "y": 1,
        }
    ]

    model = Tf2SandwichModel({
        'num_epochs': 4,
        'batch_size': 2,
        'layers': ['ggnn', 'rnn', 'ggnn'],
        'base': {
            'hidden_dim': 4,
        }
    }, num_types=3)
    model.train(data, data)
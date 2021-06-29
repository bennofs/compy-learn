import numpy as np
import tensorflow as tf
import numpy.testing as npt
import networkx as nx
import tempfile

from compy.representations.common import Graph
from compy.representations.sequence_graph import SequenceGraph
from .tf2_sandwich_model import GGNNLayer, GlobalAttentionLayer, RNNLayer, Tf2SandwichModel, \
    ragged_graph_to_leaf_sequence, segment_softmax, gather_dense_grad

tf.compat.v1.enable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True) # turn on for debugging


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
        [0, 0, 5],
        [0, 6, 1],
        [0, 8, 4],
        [1, 2, 3]
    ], dtype=tf.int32)

    out = layer(states, edges, True)


def test_global_attention_layer():
    layer = GlobalAttentionLayer()
    inputs = tf.random.uniform([10, 4], seed=0)
    graph_ids = tf.constant([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    out = layer(inputs, graph_ids)

    out_gate0 = tf.squeeze(layer.gate_layer(inputs[:2]), -1)
    out_gate0 = tf.nn.softmax(out_gate0)
    out_v0 = layer.output_layer(inputs[:2])

    npt.assert_allclose(out[0].numpy(), (out_gate0[0] * out_v0[0] + out_gate0[1] * out_v0[1]).numpy(), rtol=1e-3)


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
    out = layer(inputs, node_positions, graph_sizes)

    # non-leaf states should stay unchanged
    npt.assert_allclose(tf.boolean_mask(out, ~leaf_mask).numpy(), tf.boolean_mask(inputs, ~leaf_mask).numpy())

    layer(tf.random.uniform([1, 4], seed=0), tf.constant([1]), tf.constant([1]))


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


def test_segment_softmax():
    values = tf.random.uniform((8,), seed=213)
    sizes = tf.constant([4, 2, 2])
    segments = tf.repeat(tf.range(3), sizes)
    expected = tf.concat([
        tf.nn.softmax(values[0:4]),
        tf.nn.softmax(values[4:6]),
        tf.nn.softmax(values[6:8]),
    ], axis=0)
    npt.assert_allclose(segment_softmax(values, segments, 3).numpy(), expected.numpy(), rtol=1e-4)


def test_gather_dense_grad():
    values = tf.range(64, dtype=tf.float32)
    indices = tf.random.uniform((60,), seed=214, dtype=tf.int32, maxval=64)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(values)
        out_dense = gather_dense_grad(values, indices)
        out_sparse = tf.gather(values, indices)
        npt.assert_equal(out_dense.numpy(), out_sparse.numpy())

    grad_dense = tape.gradient(out_dense, values)
    grad_sparse = tf.convert_to_tensor(tape.gradient(out_sparse, values))
    npt.assert_equal(grad_dense.numpy(), grad_sparse.numpy(), str(indices.numpy()))


def test_train_model():
    dummy_graph = nx.MultiDiGraph()
    dummy_graph.add_node("n1", attr="a", seq_order=0)
    dummy_graph.add_node("n2", attr="b")
    dummy_graph.add_node("n3", attr="c")
    dummy_graph.add_edge("n1", "n2", attr="dummy")
    data = [
        {
            "x": {
                "code_rep": SequenceGraph.from_graph(Graph(dummy_graph, ["a", "b", "c"], ["dummy"])),
                "aux_in": [],
            },
            "y": 0,
        }
    ]

    model = Tf2SandwichModel({
        'num_epochs': 4,
        'layers': ['rnn', 'ggnn', 'rnn'],
        'base': {
            'hidden_dim': 4,
        }
    }, num_types=3)
    model.train(data, data)


def test_save_model():
    model = Tf2SandwichModel({
        'num_epochs': 4,
        'layers': ['rnn', 'ggnn', 'rnn'],
        'base': {
            'hidden_dim': 4,
        }
    }, num_types=4).model

    model({
        'nodes': tf.range(4, dtype=tf.int32),
        'node_positions': tf.range(4, dtype=tf.int32),
        'edges': tf.constant([
            (0, 1, 1),
            (2, 1, 0),
        ], dtype=tf.int32),
        'graph_sizes': tf.constant([4], dtype=tf.int32),
    })

    with tempfile.TemporaryDirectory() as dir:
        model.save(f"{dir}/model.h5")
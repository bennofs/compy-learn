import tensorflow as tf
import numpy.testing as npt
import networkx as nx

from compy.representations.common import Graph
from .tf2_ggnn_model import GGNNLayer, GlobalAttentionLayer, GGNNModel, GnnTf2Model

tf.compat.v1.enable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)


def test_ggnn_layer():
    layer = GGNNLayer({
        'num_edge_types': 2,
        'time_steps': [3,1],
        'residuals': {1: [0]},
        'gnn_h_size': 4,
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
    graph_sizes = tf.constant([2, 4, 4])
    out = layer(inputs, graph_sizes)

    out_gate0 = tf.squeeze(layer.gate_layer(inputs[:2]), -1)
    out_gate0 = tf.nn.softmax(out_gate0)
    out_v0 = layer.output_layer(inputs[:2])

    npt.assert_allclose(out[0].numpy(), (out_gate0[0] * out_v0[0] + out_gate0[1] * out_v0[1]).numpy(), rtol=1e-3)


def test_ggnn_model():
    nodes = tf.transpose(tf.constant([
        [0, 0],
        [1, 1],
        [0, 3],
        [1, 0],
        [2, 2],
        [1, 0],
        [1, 1],
    ], dtype=tf.int32), [1, 0])
    graph_sizes = tf.constant([5, 2], dtype=tf.int32)
    edges = tf.constant([
        [1, 0, 1],
        [0, 1, 2],
        [1, 5, 6],
    ], dtype=tf.int32)

    model = GGNNModel({
        'num_edge_types': 2,
        'hidden_size_orig': 3,
        'time_steps': [3, 1],
        'residuals': {1: [0]},
        'gnn_h_size': 4,
        'add_type_bias': True,
        'dropout_rate': 0,
        'pos_enc_len': 3,
    })
    model(nodes, graph_sizes, edges)


def test_train_model():
    dummy_graph = nx.MultiDiGraph()
    dummy_graph.add_node("n1", attr="a")
    dummy_graph.add_node("n2", attr="b")
    dummy_graph.add_node("n3", attr="c")
    dummy_graph.add_edge("n1", "n2", attr="dummy")
    data = [
        {
            "x": {
                "code_rep": Graph(dummy_graph, ["a", "b", "c"], ["dummy"]),
                "aux_in": [0, 0],
            },
            "y": 0,
        }
    ]

    model = GnnTf2Model({
        'num_edge_types': 1,
        'hidden_size_orig': 3,
        'time_steps': [3, 1],
        'residuals': {1: [0]},
        'gnn_h_size': 4,
        'add_type_bias': True,
        'dropout_rate': 0,
        'pos_enc_len': 3,
        'learning_rate': 0.1,
        'num_epochs': 10,
        'batch_size': 2,
    })
    model.train(data, data)
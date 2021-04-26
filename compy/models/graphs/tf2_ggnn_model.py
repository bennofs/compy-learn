"""
This implementation is adapted from https://github.com/vhellendoorn/iclr20-great
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras

from compy.models.model import Model


class GGNNLayer(tf.keras.layers.Layer):
    def __init__(self, model_config):
        super(GGNNLayer, self).__init__()
        self.num_edge_types = model_config['num_edge_types']
        # The main GGNN configuration is provided as a list of 'time-steps', which describes how often each layer is repeated.
        # E.g., an 8-step GGNN with 4 distinct layers repeated 3 and 1 times alternatingly can represented as [3, 1, 3, 1]
        self.time_steps = model_config['time_steps']
        self.num_layers = len(self.time_steps)
        # The residuals index in the time-steps above offset by one (index 0 refers to the node embeddings).
        # They describe short-cuts formatted as receiving layer: [sending layer] entries, e.g., {1: [0], 3: [0, 1]}
        self.residuals = model_config['residuals']
        self.hidden_dim = model_config['gnn_h_size']
        self.add_type_bias = model_config['add_type_bias']
        self.dropout_rate = model_config['dropout_rate']

    def build(self, _):
        # Small util functions
        random_init = tf.random_normal_initializer(stddev=self.hidden_dim ** -0.5)

        def make_weight(name=None):
            return tf.Variable(random_init([self.hidden_dim, self.hidden_dim]), name=name)

        def make_bias(name=None):
            return tf.Variable(random_init([self.hidden_dim]), name=name)

        # Set up type-transforms and GRUs
        self.type_weights = [[make_weight('type-' + str(j) + '-' + str(i)) for i in range(self.num_edge_types)] for j in
                             range(self.num_layers)]
        self.type_biases = [[make_bias('bias-' + str(j) + '-' + str(i)) for i in range(self.num_edge_types)] for j in
                            range(self.num_layers)]
        self.rnns = [tf.keras.layers.GRUCell(self.hidden_dim) for _ in range(self.num_layers)]
        for ix, rnn in enumerate(self.rnns):
            # Initialize the GRUs input dimension based on whether any residuals will be passed in.
            if str(ix) in self.residuals:
                rnn.build(self.hidden_dim * (1 + len(self.residuals[str(ix)])))
            else:
                rnn.build(self.hidden_dim)

    # Assume 'inputs' is an embedded sequence of node states, 'edge_ids' is a sparse list of indices formatted as: [edge_type, source_index, target_index].
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 3), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.bool)])
    def call(self, states, edge_ids, training):
        # Collect some basic details about the graphs in the batch.
        edge_type_ids = tf.dynamic_partition(edge_ids[:, 1:], edge_ids[:, 0], self.num_edge_types)
        message_sources = [type_ids[:, 0] for type_ids in edge_type_ids]
        message_targets = [type_ids[:, 1] for type_ids in edge_type_ids]

        # Initialize the node_states with embeddings; then, propagate through layers and number of time steps for each layer.
        layer_states = [states]
        for layer_no, steps in enumerate(self.time_steps):
            for step in range(steps):
                if str(layer_no) in self.residuals:
                    residuals = [layer_states[ix] for ix in self.residuals[str(layer_no)]]
                else:
                    residuals = None
                new_states = self.propagate(layer_states[-1], layer_no, edge_type_ids, message_sources, message_targets,
                                            residuals=residuals)
                if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
                # Add or overwrite states for this layer number, depending on the step.
                if step == 0:
                    layer_states.append(new_states)
                else:
                    layer_states[-1] = new_states
        # Return the final layer state.
        return layer_states[-1]

    def propagate(self, in_states, layer_no, edge_type_ids, message_sources, message_targets, residuals=None):
        # Collect messages across all edge types.
        messages = tf.zeros_like(in_states)
        for type_index in range(self.num_edge_types):
            type_ids = edge_type_ids[type_index]
            if tf.shape(type_ids)[0] == 0:
                continue
            # Retrieve source states and compute type-transformation.
            edge_source_states = tf.gather(in_states, message_sources[type_index])
            type_messages = tf.matmul(edge_source_states, self.type_weights[layer_no][type_index])
            if self.add_type_bias:
                type_messages += self.type_biases[layer_no][type_index]
            messages = tf.tensor_scatter_nd_add(messages, tf.expand_dims(message_targets[type_index], -1),
                                                type_messages)

        # Concatenate residual messages, if applicable.
        if residuals is not None:
            messages = tf.concat(residuals + [messages], axis=2)

        # Run GRU for each node.
        new_states, _ = self.rnns[layer_no](messages, in_states)
        return new_states


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class NodeEmbeddingLayer(tensorflow.keras.layers.Layer):
    """Computes the initial mapping from node type + position to embedding of the hidden dimension"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config["gnn_h_size"]
        self.num_classes = config["hidden_size_orig"]
        self.embed = tf.keras.layers.Embedding(self.num_classes, self.hidden_size)

        self.pos_enc = None
        if "pos_enc_len" in config:
            length = config["pos_enc_len"]
            self.pos_enc = tf.concat(
                [tf.zeros((1, self.hidden_size)), positional_encoding(self.hidden_size, length)],
                axis=0
            )

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, None), dtype=tf.int32)])
    def call(self, nodes):
        states = self.embed(nodes[0])
        if self.pos_enc is not None:
            clipped = tf.minimum(nodes[1], len(self.pos_enc) - 1)
            states += tf.gather(self.pos_enc, clipped)
        return states


@tf.function(input_signature=[tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,), dtype=tf.int32)])
def ragged_softmax(values, sizes):
    values = values - tf.reduce_max(values)  # softmax(a+c) = softmax(a), improves numerical stability
    values = tf.exp(values)
    values = tf.RaggedTensor.from_row_lengths(values, sizes)

    # compute softmax
    sums = tf.expand_dims(tf.reduce_sum(values, axis=-1), -1)
    return (values / (sums + 1e-16)).flat_values


class GlobalAttentionLayer(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate_layer = tf.keras.layers.Dense(1, activation=None)
        self.output_layer = tf.keras.layers.Dense(2, activation=None)

    def build(self, input_shape):
        self.gate_layer.build(input_shape)
        self.output_layer.build(input_shape)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None,), dtype=tf.int32)])
    def call(self, inputs, graph_sizes):
        gate = tf.squeeze(self.gate_layer(inputs), -1)
        gate = ragged_softmax(gate, graph_sizes)

        outputs = self.output_layer(inputs)
        outputs = tf.einsum('ij,i->ij', outputs, gate)
        return tf.reduce_sum(tf.RaggedTensor.from_row_lengths(outputs, graph_sizes), axis=1)


class GGNNModel(tensorflow.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.embedding_layer = NodeEmbeddingLayer(config)
        self.ggnn_layer = GGNNLayer(config)
        self.prediction_layer = GlobalAttentionLayer()

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, None)), tf.TensorSpec(shape=(None,), dtype=tf.uint32),
                                  tf.TensorSpec(shape=(None, 4))])
    def call(self, nodes, graph_sizes, edges, training=None, mask=None):
        embedded = self.embedding_layer(nodes, training=training)
        states = self.ggnn_layer(embedded, edges, training=training)
        return self.prediction_layer(states, graph_sizes, training=training)


class GnnTf2Model(Model):
    def __init__(self, config=None, num_types=None):
        if not config:
            config = {
                "timesteps": [3, 1],
                "hidden_size_orig": num_types,
                "num_edge_types": 4,
                "gnn_h_size": 32,
                "gnn_m_size": 2,
                "learning_rate": 0.001,
                "batch_size": 64,
                "num_epochs": 1000,
            }

        super().__init__(config)

        self.model = GGNNModel(config)

    @staticmethod
    def __process_data(samples):
        return [
            {
                "nodes": sample["x"]["code_rep"].get_node_list(),
                "node_positions": sample["x"]["code_rep"].get_leaf_node_positions(),
                "edges": sample["x"]["code_rep"].get_edge_list(),
                "aux_in": sample["x"]["aux_in"],
                "label": sample["y"],
            }
            for sample in samples
        ]

    def _train_init(self, data_train, data_valid):
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"])
        return self.__process_data(data_train), self.__process_data(data_valid)

    def __batch_graphs(self, batch):
        nodes = []
        edge_tensors = []
        num_nodes_so_far = 0

        for idx, graph in enumerate(batch):
            batch_nodes = np.array([
                graph["nodes"],
                [0 if p is None else 1 + p for p in graph["node_positions"]],
            ], dtype=np.int32)
            nodes.append(batch_nodes)

            batch_edges = np.array([
                (typ, idx, s + num_nodes_so_far, t + num_nodes_so_far)
                for s, typ, t in graph["edges"]
            ], dtype=np.int32)
            edge_tensors.append(batch_edges)
            num_nodes_so_far += len(graph["nodes"])

        return (
            tf.constant(np.concatenate(nodes, axis=0)),
            tf.constant([len(g["nodes"]) for g in batch], dtype=tf.int32),
            tf.constant(np.concatenate(edge_tensors, axis=0))
        )

    def _train_with_batch(self, batch):
        nodes, graph_sizes, edges = self.__batch_graphs(batch)
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        y_true = tf.constant([g['label'] for g in batch])
        with tf.GradientTape() as tape:
            y = self.model(nodes, graph_sizes, edges, training=True)
            loss = loss_func(y_true, y)
        grads = tape.gradient(loss, self.model.trainable_variables)
        accuracy.update_state(y_true, y)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, accuracy.result().numpy()

    def _predict_with_batch(self, batch):
        nodes, graph_sizes, edges = self.__batch_graphs(batch)
        y_true = tf.constant([g['label'] for g in batch])
        y = self.model(nodes, graph_sizes, edges, training=False)
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        accuracy.update_state(y_true, y)
        return accuracy.result().numpy(), y

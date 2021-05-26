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
        # The main GGNN configuration is provided as a list of 'time-steps', which describes how often each layer is
        # repeated. E.g., an 8-step GGNN with 4 distinct layers repeated 3 and 1 times alternatingly can represented
        # as [3, 1, 3, 1]
        self.time_steps = model_config['time_steps']
        self.num_layers = len(self.time_steps)
        # The residuals index in the time-steps above offset by one (index 0 refers to the node embeddings).
        # They describe short-cuts formatted as receiving layer: [sending layer] entries, e.g., {1: [0], 3: [0, 1]}
        self.residuals = model_config['residuals']
        self.hidden_dim = model_config['hidden_dim']
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

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, edge_ids, training=True, **kwargs):
        """Run the GGNN layer on the given states for the graph specified by edge_ids."""
        # Collect some basic details about the graphs in the batch.
        edge_type_ids = tf.dynamic_partition(edge_ids[:, 1:], edge_ids[:, 0], self.num_edge_types)
        message_sources = [type_ids[:, 0] for type_ids in edge_type_ids]
        message_targets = [type_ids[:, 1] for type_ids in edge_type_ids]

        # Initialize the node_states with embeddings
        # then, propagate through layers and number of time steps for each layer.
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
            messages = tf.concat(residuals + [messages], axis=-1)

        # Run GRU for each node.
        new_states, _ = self.rnns[layer_no](messages, in_states)
        return new_states


def positional_encoding(dim: int, sentence_length: int, dtype=tf.float32):
    """Compute positional encodings for all positions in sequences of given maximum length.

    :param dim: Dimension of each positional encoding
    :param sentence_length: Number of positional encodings to compute (maximum number of positions)
    :param dtype: Dtype of the positional encoding vectors
    """
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class NodeEmbeddingLayer(tensorflow.keras.layers.Layer):
    """Compute the initial mapping from node type + position to embedding of the hidden dimension"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config["hidden_dim"]
        self.num_classes = config["hidden_size_orig"]
        self.embed = tf.keras.layers.Embedding(self.num_classes, self.hidden_size)

        self.pos_enc = None
        if "pos_enc_len" in config:
            length = config["pos_enc_len"]
            self.pos_enc = tf.concat(
                [tf.zeros((1, self.hidden_size)), positional_encoding(self.hidden_size, length)],
                axis=0
            )

    @tf.function(experimental_relax_shapes=True)
    def call(self, nodes, node_positions):
        states = self.embed(nodes)
        if self.pos_enc is not None:
            clipped = tf.minimum(node_positions, len(self.pos_enc) - 1)
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

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, graph_sizes, training=True):
        gate = tf.squeeze(self.gate_layer(states), -1)
        gate = ragged_softmax(gate, graph_sizes)

        outputs = self.output_layer(states)
        outputs = tf.einsum('ij,i->ij', outputs, gate)
        return tf.reduce_sum(tf.RaggedTensor.from_row_lengths(outputs, graph_sizes), axis=1)


def ragged_graph_to_leaf_sequence(nodes, node_positions):
    leaf_mask = tf.not_equal(node_positions, 0)
    leaf_positions = tf.ragged.boolean_mask(node_positions, leaf_mask)
    leaf_nodes = tf.ragged.boolean_mask(nodes, leaf_mask)

    leaf_seqstarts = tf.cast(
        tf.expand_dims(leaf_positions.row_splits[:-1], axis=-1),
        dtype=leaf_positions.dtype,
    )

    flat_indices = tf.expand_dims((leaf_positions - 1) + leaf_seqstarts, axis=-1)
    return tf.ragged.map_flat_values(
        tf.scatter_nd,
        flat_indices,
        leaf_nodes,
        shape=leaf_nodes.flat_values.shape
    )


class RNNLayer(tf.keras.layers.Layer):
    def __init__(self, model_config):
        super(RNNLayer, self).__init__()
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.dropout_rate = model_config['dropout_rate']

        self.rnns_bwd = [tf.keras.layers.GRU(self.hidden_dim // 2, return_sequences=True, go_backwards=True) for _ in
                         range(self.num_layers)]
        self.rnns_fwd = [tf.keras.layers.GRU(self.hidden_dim // 2, return_sequences=True) for _ in
                         range(self.num_layers)]

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, node_positions, graph_sizes, training=True):
        # Extract sequence from graph leaves
        node_positions = tf.RaggedTensor.from_row_lengths(node_positions, graph_sizes)
        states = tf.RaggedTensor.from_row_lengths(states, graph_sizes)
        sequence = ragged_graph_to_leaf_sequence(states, node_positions)

        # Run sequence through all layers.
        real_dropout_rate = self.dropout_rate * tf.cast(training, 'float32')
        for layer_no in range(self.num_layers):
            fwd = self.rnns_fwd[layer_no](sequence)
            bwd = self.rnns_bwd[layer_no](sequence)
            sequence = tf.concat([fwd, bwd], axis=-1)
            sequence = tf.ragged.map_flat_values(tf.nn.dropout, sequence, rate=real_dropout_rate)

        # Scatter sequence back into graph states
        orig_indices = tf.range(tf.shape(states.flat_values)[0], dtype=tf.int32)
        orig_indices = tf.RaggedTensor.from_row_lengths(orig_indices, graph_sizes)
        dest_indices = ragged_graph_to_leaf_sequence(orig_indices, node_positions)
        dest_indices = tf.expand_dims(dest_indices.flat_values, axis=-1)

        return tf.tensor_scatter_nd_update(states.flat_values, dest_indices, sequence.flat_values)


class SandwichModel(tf.keras.Model):
    LAYER_CLASSES = {'rnn': RNNLayer, 'ggnn': GGNNLayer}

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        layer_config = {}
        for layer in ('embed', 'rnn', 'ggnn'):
            layer_config[layer] = dict(config['base'], **config.get(layer, {}))

        self.embed = NodeEmbeddingLayer(layer_config['embed'])
        self.attention = GlobalAttentionLayer()

        self.stack = []
        for layer in config['layers']:
            self.stack += [(layer, SandwichModel.LAYER_CLASSES[layer](layer_config[layer]))]

    @tf.function(experimental_relax_shapes=True)
    def call(self, nodes, edges, node_positions, graph_sizes, training=True, mask=None):
        states = self.embed(nodes, node_positions)

        for kind, layer in self.stack:
            if kind == 'ggnn':
                states = layer(states, edges)
            if kind == 'rnn':
                states = layer(states, node_positions, graph_sizes)

        return self.attention(states, graph_sizes)


class Tf2SandwichModel(Model):

    def __init__(self, config=None, num_types=None, num_edge_types=4):
        config = {} if config is None else config
        base_config = config.get('base', {})
        ggnn_config = config.get('ggnn', {})
        rnn_config = config.get('rnn', {})

        config = {
            'layers': config.get('layers', ['rnn', 'ggnn', 'rnn', 'ggnn', 'rnn']),
            "learning_rate": config.get('learning_rate', 0.001),
            "batch_size": config.get('batch_size', 64),
            "num_epochs": config.get('num_epochs', 1000),

            'base': {
                "num_edge_types": base_config.get('num_edge_types', num_edge_types),
                'hidden_size_orig': base_config.get('hidden_size_orig', num_types),
                'hidden_dim': base_config.get('hidden_dim', 512),
                "dropout_rate": base_config.get('dropout_rate', 0.1),
            },

            'ggnn': {
                'time_steps': ggnn_config.get('time_steps', [3, 1]),
                'residuals': ggnn_config.get('residuals', {
                    '1': [0],
                }),
                'add_type_bias': ggnn_config.get('add_type_bias', True),
            },

            'rnn': {
                'num_layers': rnn_config.get('num_layers', 2),
            },

            'embed': {
            }
        }

        super().__init__(config)
        self.model = SandwichModel(config)


    @staticmethod
    def __process_data(samples):
        return [
            {
                "nodes": sample["x"]["code_rep"].nodes,
                "node_positions": sample['x']['code_rep'].get_node_positions(),
                "edges": sample["x"]["code_rep"].edges,
                "label": sample["y"],
            }
            for sample in samples
        ]

    def _train_init(self, data_train, data_valid):
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"])
        return self.__process_data(data_train), self.__process_data(data_valid)

    @staticmethod
    def __batch_graphs(batch):
        nodes = []
        node_positions = []
        edge_tensors = []
        num_nodes_so_far = 0

        for idx, graph in enumerate(batch):
            batch_nodes = np.array(graph['nodes'], dtype=np.int32)
            batch_node_positions = np.array(
                [0 if p is None else 1 + p for p in graph["node_positions"]],
                dtype=np.int32)
            nodes.append(batch_nodes)
            node_positions.append(batch_node_positions)

            batch_edges = np.array([
                (typ, idx, s + num_nodes_so_far, t + num_nodes_so_far)
                for typ, s, t in zip(*graph["edges"])
            ], dtype=np.int32)
            edge_tensors.append(batch_edges)
            num_nodes_so_far += len(graph["nodes"])

        return {
            'nodes': tf.constant(np.concatenate(nodes, axis=0)),
            'node_positions': tf.constant(np.concatenate(node_positions, axis=0)),
            'graph_sizes': tf.constant([len(g["nodes"]) for g in batch], dtype=tf.int32),
            'edges': tf.constant(np.concatenate(edge_tensors, axis=0))
        }

    def _train_with_batch(self, batch):
        x = self.__batch_graphs(batch)
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        y_true = tf.constant([g['label'] for g in batch])

        with tf.GradientTape() as tape:
            y = self.model(**x, training=True)
            loss = loss_func(y_true, y)
        grads = tape.gradient(loss, self.model.trainable_variables)
        accuracy.update_state(y_true, y)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, accuracy.result().numpy()

    def _predict_with_batch(self, batch):
        x = self.__batch_graphs(batch)
        y_true = tf.constant([g['label'] for g in batch])
        y = self.model(**x, training=False)

        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        accuracy.update_state(y_true, y)
        return accuracy.result().numpy(), y

"""
This implementation is adapted from https://github.com/vhellendoorn/iclr20-great
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras

from compy.models.model import Model


@tf.function(autograph=False)
@tf.custom_gradient
def gather_dense_grad(params, indices):
    """Like tf.gather but with a dense gradient.

    ``tf.gather(params, indices)`` has a sparse gradient: only the selected elements from params
    affect the output, so the gradient of ``params`` is zero at all positions that are not selected by ``indices``.
    This is useful of ``params`` is very large and ``indices`` only selects a small subset of it.

    However, sometimes we know that ``params`` is not too large and we need an explicit gradient for each element
    anyway. In this case, it doesn't make sense to compute a sparse gradient first and then convert it to a dense
    representation later. Thus, this version of ``tf.gather`` directly computes a dense gradient.
    """
    grad_shape = tf.shape(params)

    def grad(dy):
        return tf.tensor_scatter_nd_add(tf.zeros(grad_shape), indices[:, tf.newaxis], dy), None

    return tf.gather(params, indices), grad


class GGNNLayer(tf.keras.layers.Layer):
    def __init__(self, model_config, **kwargs):
        super(GGNNLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self._model_config = model_config

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

    def get_config(self):
        base_config = super(GGNNLayer, self).get_config()
        return dict(base_config, model_config=self._model_config)

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
        self.rnns = [tf.keras.layers.GRUCell(self.hidden_dim) for i in range(self.num_layers)]

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, edges, training=True, **kwargs):
        """Run the GGNN layer on the given states for the graph specified by ``edges``."""
        # Collect some basic details about the graphs in the batch.
        edge_type_ids = tf.dynamic_partition(edges[:, 1:], edges[:, 0], self.num_edge_types)
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
                                            training, residuals=residuals)
                if training: new_states = tf.nn.dropout(new_states, rate=self.dropout_rate)
                # Add or overwrite states for this layer number, depending on the step.
                if step == 0:
                    layer_states.append(new_states)
                else:
                    layer_states[-1] = new_states
        # Return the final layer state.
        return layer_states[-1]

    def propagate(self, in_states, layer_no, edge_type_ids, message_sources, message_targets, training, residuals=None):
        # Collect messages across all edge types.
        messages = tf.zeros_like(in_states)
        for type_index in range(self.num_edge_types):
            type_ids = edge_type_ids[type_index]
            if tf.shape(type_ids)[0] == 0:
                continue
            # Retrieve source states and compute type-transformation.
            edge_source_states = gather_dense_grad(in_states, message_sources[type_index])
            type_messages = tf.matmul(edge_source_states, self.type_weights[layer_no][type_index])
            if self.add_type_bias:
                type_messages += self.type_biases[layer_no][type_index]
            messages = tf.tensor_scatter_nd_add(messages, message_targets[type_index][..., tf.newaxis], type_messages)

        # Concatenate residual messages, if applicable.
        if residuals is not None:
            messages = tf.concat(residuals + [messages], axis=-1)

        # Run GRU for each node.
        new_states, _ = self.rnns[layer_no](messages, in_states, training=training)
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
        self._config = config
        self.hidden_size = config["hidden_dim"]
        self.num_classes = config["hidden_size_orig"]
        self.embed = tf.keras.layers.Embedding(self.num_classes, self.hidden_size)
        self.supports_masking = True

        self.pos_enc = None
        if "pos_enc_len" in config:
            length = config["pos_enc_len"]
            self.pos_enc = tf.concat(
                [tf.zeros((1, self.hidden_size)), positional_encoding(self.hidden_size, length)],
                axis=0
            )

    def get_config(self):
        base_config = super(NodeEmbeddingLayer, self).get_config()
        return dict(base_config, config=self._config)

    @tf.function(experimental_relax_shapes=True)
    def call(self, nodes, node_positions):
        states = self.embed(nodes)
        if self.pos_enc is not None:
            clipped = tf.minimum(node_positions, len(self.pos_enc) - 1)
            states += tf.gather(self.pos_enc, clipped)

        return states


def segment_softmax(values, segments, num_segments, mask=None):
    # softmax(a+c) = softmax(a), improves numerical stability
    # don't propagate gradient through the max, just treat it as constant
    values = values - tf.stop_gradient(tf.reduce_max(values))
    values = tf.exp(values)

    if mask is not None:
        values = values * tf.cast(mask, dtype=values.dtype)

    # compute softmax
    sums = tf.math.unsorted_segment_sum(values, segments, num_segments)
    sums = gather_dense_grad(sums, segments)
    return values / (sums + 1e-16)


class GlobalAttentionLayer(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate_layer = tf.keras.layers.Dense(1, activation=None)
        self.output_layer = tf.keras.layers.Dense(2, activation=None)
        self.supports_masking = False

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, graph_ids, mask=None):
        num_graphs = tf.reduce_max(graph_ids) + 1

        gate = tf.squeeze(self.gate_layer(states), -1)
        gate = segment_softmax(gate, graph_ids, num_graphs, mask=mask)

        outputs = self.output_layer(states)
        outputs = tf.einsum('ij,i->ij', outputs, gate)
        return tf.math.unsorted_segment_sum(outputs, graph_ids, num_graphs)


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
        shape=tf.shape(leaf_nodes.flat_values)
    )


class RNNLayer(tf.keras.layers.Layer):
    def __init__(self, model_config):
        super(RNNLayer, self).__init__()
        self._model_config = model_config
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.dropout_rate = model_config['dropout_rate']

        self.rnns = [
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dim // 2, return_sequences=True))
            for _ in range(self.num_layers)
        ]

    def get_config(self):
        base_config = super(RNNLayer, self).get_config()
        return dict(base_config, model_config=self._model_config)

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, node_positions, graph_sizes, training=True):
        # Extract sequence from graph leaves
        node_positions = tf.RaggedTensor.from_row_lengths(node_positions, graph_sizes)
        states = tf.RaggedTensor.from_row_lengths(states, graph_sizes)
        sequence = ragged_graph_to_leaf_sequence(states, node_positions)

        # Run sequence through all layers.
        real_dropout_rate = self.dropout_rate * tf.cast(training, 'float32')
        for layer_no in range(self.num_layers):
            sequence = self.rnns[layer_no](sequence)
            sequence = tf.ragged.map_flat_values(tf.nn.dropout, sequence, rate=real_dropout_rate)

        # Scatter sequence back into graph states
        orig_indices = tf.range(tf.shape(states.flat_values)[0], dtype=tf.int32)
        orig_indices = tf.RaggedTensor.from_row_lengths(orig_indices, graph_sizes)
        dest_indices = ragged_graph_to_leaf_sequence(orig_indices, node_positions)
        dest_indices = tf.expand_dims(dest_indices.flat_values, axis=-1)

        return tf.tensor_scatter_nd_update(states.flat_values, dest_indices, sequence.flat_values)


class DenseRNNLayer(tf.keras.layers.Layer):
    def __init__(self, model_config, **kwargs):
        super(DenseRNNLayer, self).__init__(**kwargs)
        self._model_config = model_config
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.dropout_rate = model_config['dropout_rate']

        self.rnns = [
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dim // 2, return_sequences=True))
            for _ in range(self.num_layers)
        ]
        self.supports_masking = True

    def build(self, input_shape):
        # if we don't call build explicitly here, then RNNs won't use the optimized cuDNN kernels for some reason
        # not using the optimized cuDNN kernel has a 60x performance penalty
        for rnn in self.rnns:
            rnn.build((None, None, self.hidden_dim))

    def get_config(self):
        base_config = super(DenseRNNLayer, self).get_config()
        return dict(base_config, model_config=self._model_config)

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, seq_shape, training=True, mask=None):
        assert len(seq_shape) == 2, "seq_shape must be 2D: (num_sequences, length_per_sequence)"
        total_len = seq_shape[0] * seq_shape[1]
        sequence = tf.reshape(states[:total_len], tf.concat([seq_shape, (self.hidden_dim,)], axis=0))
        mask = tf.reshape(mask[:total_len], seq_shape)

        real_dropout_rate = self.dropout_rate * tf.cast(training, 'float32')
        for rnn in self.rnns:
            sequence = rnn(sequence, mask=mask)
            sequence = tf.nn.dropout(sequence, rate=real_dropout_rate)

        return tf.concat([tf.reshape(sequence, (-1, self.hidden_dim)), states[total_len:]], axis=0)


def pack_ragged_batch_to_dense_input(*, seq_nodes: tf.RaggedTensor, non_seq_nodes: tf.RaggedTensor, edges: tf.RaggedTensor):
    """
    Rake a batch of train inputs as ragged tensors (sequence lengths and number of nodes can differ for each sample)
    and pack it into a dense representation to be used as input for a sandwich model with rnn_dense=True.

    :param seq_nodes: The sequence nodes for each sample
    :param non_seq_nodes: The internal (non-sequence) nodes for each sample
    :param edges: Edges for each sample, as (type, src, dest), where src and dest are indicies into the
                  concatenation of seq_nodes and non_seq_nodes for the sample.
    :return: A dict that can be passed as input to a sandwich model with rnn_dense=True
    """
    # we want to pack the node tensor as follows:
    # [s1 s1 s1 0 0 s2 s2 s2 s2 s2 s3 s3 0 0 0 n1 n1 n1 n2 n2 n3]
    # where s1, s2 and s3 are sequence nodes of the first, second and third sample
    # and n1, n2 and n3 are non-sequence nodes of the first, second and third sample

    # we first pad the ragged sequence nodes, to get a rectangular shape:
    # [s1 s1 s1  0  0]
    # [s2 s2 s2 s2 s2]
    # [s3 s3 0   0  0]
    seq_nodes_padded = seq_nodes.to_tensor()
    seq_shape = tf.shape(seq_nodes_padded)

    # compute the mask that is 0 for the added padding, and 1 otherwise:
    # [ 1  1  1  0  0]
    # [ 1  1  1  1  1]
    # [ 1  1  0  0  0]
    # and reshape it to be flat 1d:
    # [ 1 1 1 0 0  1 1 1 1 1  1 1 0 0 0]
    seq_mask = tf.reshape(tf.ones_like(seq_nodes, dtype=tf.bool).to_tensor(), (-1,))

    # now pack the nodes by concatenating the flat 1d padded sequence nodes and the unpadded non-sequence nodes
    num_seq_nodes = seq_shape[0] * seq_shape[1]
    num_non_seq_nodes = tf.shape(non_seq_nodes.flat_values)[0]
    nodes = tf.concat([tf.reshape(seq_nodes_padded, (-1, )), non_seq_nodes.flat_values], axis=0)

    # seq_positions is a flat tensor of indicies for each element in the sequence (1-based):
    # [1 2 3 4 5}
    # [1 2 3 4 5]
    # [1 2 3 4 5]
    # we then mask this tensor to zero the indices for padded sequence locations and pad it with zeros for the
    # non-sequence nodes, so we get the node_positions:
    # [1 2 3 0 0 1 2 3 4 5 1 2 0 0 0 0 0 0 0 0 0]
    seq_len = tf.cast(seq_nodes.row_lengths(), edges.dtype),
    seq_positions = tf.reshape(
        tf.repeat(tf.range(seq_shape[1], dtype=tf.int32)[tf.newaxis, :], seq_shape[0], axis=0),
        (-1,)
    ) + 1
    node_positions = tf.pad(seq_positions * tf.cast(seq_mask, seq_positions.dtype), [[0, num_non_seq_nodes]])

    # seq_offset is the index of the first sequence node of the graph associated with each edge
    # if we have 2 edges in the first sample, 1 in the second and 3 in the third, then seq_offsets is (for the nodes
    # given in the examples before):
    # [0 0 5 10 10 10]
    # because the sequence nodes start at index 0 for the first sample, 5 for the second and 10 for the third
    seq_offset = tf.range(seq_shape[0]) * seq_shape[1]
    seq_offset = tf.repeat(seq_offset, edges.row_lengths())

    # non_seq_offset is the index of the first non-sequence node of the graph associated with each edge
    # for our example, the row starts of non-sequential nodes are [0, 3, 5]
    # we need to offset those by the number of padded sequence nodes to get the starts in the packed nodes tensor:
    # (since all the sequence nodes are before the first non-seq node in the packed node tensor): [15, 18, 20]
    # and then we repeat them for each edge: [15 15 18 20 20 20]
    non_seq_starts = tf.cast(non_seq_nodes.row_starts(), edges.dtype)
    non_seq_offset = non_seq_starts + num_seq_nodes - seq_len
    non_seq_offset = tf.repeat(non_seq_offset, edges.row_lengths())

    # for each edge, the number of sequence nodes in the associated graph
    seq_len_edges = tf.repeat(seq_len, edges.row_lengths())

    # transform edge indices to indices into the packed node tensor
    def offset_edges(x):
        is_seq = tf.cast(tf.less(x, seq_len_edges), x.dtype)
        return x + seq_offset * is_seq + non_seq_offset * (1 - is_seq)

    edge_type, edge_src, edge_dst = tf.unstack(edges.flat_values, axis=-1)
    edges = tf.stack([edge_type, offset_edges(edge_src), offset_edges(edge_dst)], axis=-1)

    num_nodes = tf.shape(nodes)[0]
    tf.assert_less(edges[:, 1], num_nodes)
    tf.assert_less(edges[:, 2], num_nodes)

    # graph_ids contains the sample id for each node in the flat node tensor
    num_graphs = seq_shape[0]
    seq_graph_ids = tf.reshape(tf.broadcast_to(tf.range(num_graphs)[:, tf.newaxis], seq_shape), (-1, ))
    non_seq_graph_ids = tf.repeat(tf.range(num_graphs), non_seq_nodes.row_lengths())
    graph_ids = tf.concat([seq_graph_ids, non_seq_graph_ids], axis=0)

    return {
        'edges': edges,
        'nodes': nodes,
        'graph_ids': graph_ids,
        'mask': tf.concat([tf.reshape(seq_mask, (num_seq_nodes, )), tf.ones(num_non_seq_nodes, dtype=tf.bool)], axis=0),
        'seq_shape': seq_shape,
        'node_positions': node_positions,
    }


def sandwich_model(config, rnn_dense=False):
    layer_config = {}
    for layer in ('embed', 'rnn', 'ggnn'):
        layer_config[layer] = dict(config['base'], **config.get(layer, {}))

    # inputs
    nodes = tf.keras.Input(shape=(), dtype=tf.int32, name="nodes")
    node_positions = tf.keras.Input(shape=(), dtype=tf.int32, name="node_positions")
    seq_shape = tf.keras.Input(shape=(), batch_size=2, dtype=tf.int32, name="seq_shape")
    seq_mask = tf.keras.Input(shape=(), dtype=tf.bool, name="mask")
    edges = tf.keras.Input(shape=(3,), dtype=tf.int32, name="edges")
    graph_sizes = tf.keras.Input(shape=(), dtype=tf.int32, name="graph_sizes")

    # compute graph embeddings
    states = NodeEmbeddingLayer(layer_config['embed'])(nodes=nodes, node_positions=node_positions)
    for layer_kind in config['layers']:
        if layer_kind == 'rnn':
            if rnn_dense:
                states = DenseRNNLayer(layer_config['rnn'])(states=states, seq_shape=seq_shape, mask=seq_mask)
            else:
                states = RNNLayer(layer_config['rnn'])(
                    states=states, node_positions=node_positions, graph_sizes=graph_sizes
                )
            continue

        if layer_kind == 'ggnn':
            states = GGNNLayer(layer_config['ggnn'])(states=states, edges=edges)
            continue


        raise ValueError("unknown model layer type: " + layer_kind + ", expected one of [ggnn, rnn, rnn-dense]")

    # outputs
    if rnn_dense:
        graph_ids = tf.keras.Input(shape=(), dtype=tf.int32, name="graph_ids")
    else:
        num_graphs = tf.shape(graph_sizes)[0]
        graph_ids = tf.repeat(tf.range(num_graphs, dtype=tf.int32), graph_sizes)
    output = GlobalAttentionLayer()(states=states, graph_ids=graph_ids)
    output = tf.nn.softmax(output)

    inputs = (nodes, node_positions, seq_shape, seq_mask, edges, graph_ids) if rnn_dense else (
        nodes, node_positions, edges, graph_sizes)
    return tf.keras.Model(inputs=inputs, outputs=output)


class Tf2SandwichModel(Model):

    def __init__(self, config=None, num_types=None, num_edge_types=4, keras_callbacks=None):
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

        self.model = sandwich_model(config)
        self._callbacks = tf.keras.callbacks.CallbackList(keras_callbacks, model=self.model)
        self._step = 0
        self._epoch = 0

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

    def _epoch_init(self, epoch):
        if epoch > 0:
            self._callbacks.on_epoch_end(epoch - 1)

        self._epoch = epoch
        self._step = 0

        self._callbacks.on_epoch_begin(self._epoch)

    def _train_init(self, data_train, data_valid):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        self._callbacks.on_train_begin()

        return self.__process_data(data_train), self.__process_data(data_valid)

    def _test_init(self):
        super()._test_init()
        self._callbacks.on_test_begin()
        self._step = 0

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
                (typ, s + num_nodes_so_far, t + num_nodes_so_far)
                for typ, s, t in zip(*graph["edges"])
            ], dtype=np.int32)
            edge_tensors.append(tf.constant(batch_edges, shape=(len(batch_edges), 3)))
            num_nodes_so_far += len(graph["nodes"])

        return {
            'nodes': tf.constant(np.concatenate(nodes, axis=0)),
            'node_positions': tf.constant(np.concatenate(node_positions, axis=0)),
            'graph_sizes': tf.constant([len(g["nodes"]) for g in batch], dtype=tf.int32),
            'edges': tf.constant(np.concatenate(edge_tensors, axis=0))
        }

    def _train_with_batch(self, batch):
        with tf.profiler.experimental.Trace(
                name='train', epoch_num=self._epoch, step_num=self._step, batch_size=len(batch), _r=1):
            self._callbacks.on_train_batch_begin(self._step)

            x = self.__batch_graphs(batch)
            y_true = tf.constant([g['label'] for g in batch])

            data = tf.data.Dataset.from_tensors((x, y_true))
            iterator = iter(data)
            result = self.model.make_train_function()(iterator)
            result = {k: v.numpy() for k, v in result.items()}

            self.model.reset_metrics()

            self._callbacks.on_train_batch_end(self._step, result)

        self._step += 1

        return result['loss'], result['accuracy']

    def _predict_with_batch(self, batch):
        self._callbacks.on_predict_batch_begin(self._step)

        x = self.__batch_graphs(batch)
        y_true = tf.constant([g['label'] for g in batch])
        y = self.model(x, training=False)

        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        accuracy.update_state(y_true, y)

        self._step += 1
        self._callbacks.on_predict_batch_end(self._step, {'accuracy': accuracy})

        return accuracy.result().numpy(), y

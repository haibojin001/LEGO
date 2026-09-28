# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg41::math.sqrt+tensorflow.cast+tensorflow.expand_dims
# name: math_tensorflow_primitive
# summary: Uses math.sqrt, tensorflow.cast, tensorflow.expand_dims, tensorflow.matmul across 3 repos
# anchor_symbols: ['math.sqrt', 'tensorflow.cast', 'tensorflow.expand_dims', 'tensorflow.matmul', 'tensorflow.one_hot', 'tensorflow.ones_like']
# observed in 3 repos: ['awslabs__gluonts', 'd2l-ai__d2l-en', 'google__uncertainty-baselines']...

# --- from awslabs__gluonts::src/gluonts/nursery/robust-mts-attack/pts/modules/iqn_modules.py::QuantileLayer.cos_embed ---
def cos_embed(self, tau):
        integers = torch.repeat_interleave(
            torch.arange(0, self.n_cos_embedding).unsqueeze(dim=0),
            repeats=tau.shape[-1],
            dim=0,
        ).to(tau.device)
        return torch.cos(pi * tau.unsqueeze(dim=-1) * integers)

# --- from d2l-ai__d2l-en::d2l/tensorflow.py::DotProductAttention.call ---
def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)

# --- from d2l-ai__d2l-en::d2l/tensorflow.py::masked_softmax._sequence_mask ---
def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)

# --- from google__uncertainty-baselines::uncertainty_baselines/models/gat.py::GraphAttentionLayer._calc_attention_scores ---
def _calc_attention_scores(self, attention_inputs):
    """Compute attention scores.

    Args:
      attention_inputs: The incoming tensor contains pair-wise
        node-level representations. It has dimension of (batch_size,
        num_nodes, num_nodes, 2*out_node_feature_dim)
    Returns:
      attention_scores: Computed attention scores tensor with dimension
        of (batch_size, num_nodes, num_nodes)
    """
    attention_scores = tf.squeeze(
        tf.matmul(attention_inputs, self.a),
        axis=[3])  # (batch_size, num_nodes, num_nodes)
    attention_scores = self.leakyrelu(
        attention_scores)  # (batch_size, num_nodes, num_nodes)
    return attention_scores

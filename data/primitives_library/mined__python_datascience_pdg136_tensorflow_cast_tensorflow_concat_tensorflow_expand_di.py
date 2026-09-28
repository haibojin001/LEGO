# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg136::tensorflow.cast+tensorflow.concat+tensorflow.expand_dims
# name: tensorflow_util_primitive
# summary: Uses tensorflow.cast, tensorflow.concat, tensorflow.expand_dims, tensorflow.ones_like across 2 repos
# anchor_symbols: ['tensorflow.cast', 'tensorflow.concat', 'tensorflow.expand_dims', 'tensorflow.ones_like', 'tensorflow.shape', 'tensorflow.sqrt', 'tensorflow.tile', 'tensorflow.variable_scope', 'tensorflow.zeros', 'util.dropout']
# observed in 2 repos: ['google__uncertainty-baselines', 'microsoft__nni']...

# --- from google__uncertainty-baselines::uncertainty_baselines/optimizers_test.py::OptimizersTest.testAdamW ---
def testAdamW(self):
    weight_decay = 1e-3
    optimizer = ub.optimizers.get(
        optimizer_name='adam',
        learning_rate=0.1,
        weight_decay=weight_decay,
        learning_rate_schedule='constant',
        beta_1=0.9,
        epsilon=1e-1)
    shape = (7,)
    initial_value = tf.ones(shape)
    test_var = tf.Variable(name='test_var', initial_value=initial_value)
    zeros_update = [(tf.zeros(shape), test_var)]
    optimizer.apply_gradients(zeros_update)
    # Because we gave a gradient of 0, the only update to the variable should be
    # from the weight decay.
    self.assertAllClose(initial_value - weight_decay, test_var)

# --- from microsoft__nni::examples/trials/weight_sharing/ga_squad/graph_to_tf.py::normalize ---
def normalize(inputs,
              epsilon=1e-8,
              scope="ln"):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

# --- from microsoft__nni::examples/trials/ga_squad/graph_to_tf.py::normalize ---
def normalize(inputs,
              epsilon=1e-8,
              scope="ln"):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

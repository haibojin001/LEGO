# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg70::tensorflow.cast
# name: tensorflow_primitive
# summary: Uses tensorflow.cast across 12 repos
# anchor_symbols: ['tensorflow.cast']
# observed in 12 repos: ['Lightning-AI__torchmetrics', 'ahmetozlu__tensorflow_object_counting_api', 'capitalone__DataProfiler', 'google__uncertainty-baselines', 'iterative__mlem']...

# --- from google__uncertainty-baselines::baselines/jft/active_learning.py::main.write_note ---
def write_note(note):
    if jax.process_index() == 0:
      logging.info('NOTE: %s', note)

# --- from google__uncertainty-baselines::baselines/imagenet/deterministic.py::main.moving_average_step.step_fn_images ---
def step_fn_images(images):
      return tf.reduce_mean(tf.cast(images, tf.float32), axis=0)

# --- from lazyprogrammer__machine_learning_examples::cnn_class2/tf_resnet.py::custom_softmax ---
def custom_softmax(x):
  m = tf.reduce_max(x, 1)
  x = x - m
  e = tf.exp(x)
  return e / tf.reduce_sum(e, -1)

# --- from lazyprogrammer__machine_learning_examples::cnn_class2/test_softmax.py::custom_softmax ---
def custom_softmax(x):
  m = tf.reduce_max(x, 1)
  x = x - m
  e = tf.exp(x)
  return e / tf.reduce_sum(e, -1)

# --- from iterative__mlem::tests/contrib/test_tensorflow.py::tftt_3d ---
def tftt_3d(tensor_data):
    return DataAnalyzer.analyze(
        tf.tile(tf.expand_dims(tensor_data, -1), [1, 1, 20])
    )

# --- from microsoft__nni::examples/nas/legacy/oneshot/naive-tf/train.py::accuracy ---
def accuracy(truth, logits):
    truth = tf.reshape(truth, (-1, ))
    predicted = tf.cast(tf.math.argmax(logits, axis=1), truth.dtype)
    equal = tf.cast(predicted == truth, tf.int32)
    return tf.math.reduce_sum(equal).numpy() / equal.shape[0]

# --- from capitalone__DataProfiler::dataprofiler/labelers/labeler_utils.py::FBetaScore.update_state._weighted_sum ---
def _weighted_sum(val: tf.Tensor, sample_weight: tf.Tensor | None) -> tf.Tensor:
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

# --- from makcedward__nlp::aion/embeddings/elmo.py::ELMoEmbeddings.to_keras_layer ---
def to_keras_layer(self, x):
        # Source: https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
        '''
            For signature and layer parameters, you can visit https://alpha.tfhub.dev/google/elmo/2
        '''        
        return self.model(
            tf.squeeze(tf.cast(x, tf.string)), 
            signature="default", as_dict=True)[self.layer]

# --- from keras-team__keras-contrib::keras_contrib/backend/tensorflow_backend.py::_postprocess_conv2d_output ---
def _postprocess_conv2d_output(x, data_format):
    """Transpose and cast the output from conv2d if needed.

    # Arguments
        x: A tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """

    if data_format == 'channels_first':
        x = tf.transpose(x, (0, 3, 1, 2))

    if K.floatx() == 'float64':
        x = tf.cast(x, 'float64')
    return x

# --- from microsoft__nni::examples/trials/weight_sharing/ga_squad/attention.py::DotAttention.get_att ---
def get_att(self, s, prob):
        '''
        :param s: [src_sequence_length, batch_size, src_dim]
        :param prob: [src_sequence_length, batch_size]\
            or [tgt_sequence_length, src_sequence_length, batch_size]
        :return: [batch_size, src_dim] or [tgt_sequence_length, batch_size, src_dim]
        '''
        buf = s * tf.expand_dims(prob, axis=-1)
        att = tf.reduce_sum(buf, axis=-3)
        return att

# --- from keras-team__keras-contrib::keras_contrib/backend/tensorflow_backend.py::_preprocess_conv2d_input ---
def _preprocess_conv2d_input(x, data_format):
    """Transpose and cast the input before the conv2d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """
    if K.dtype(x) == 'float64':
        x = tf.cast(x, 'float32')
    if data_format == 'channels_first':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = tf.transpose(x, (0, 2, 3, 1))
    return x

# --- from capitalone__DataProfiler::dataprofiler/labelers/labeler_utils.py::FBetaScore.result ---
def result(self) -> tf.Tensor:
        """Return f1 score."""
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

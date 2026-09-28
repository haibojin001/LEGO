# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg573::tensorflow.cast
# name: tensorflow_primitive
# summary: Uses tensorflow.cast across 2 repos
# anchor_symbols: ['tensorflow.cast']
# observed in 2 repos: ['google__uncertainty-baselines', 'terrytangyuan__distributed-ml-patterns']...

# --- from google__uncertainty-baselines::experimental/multimodal/randaugment.py::autocontrast.scale_channel.scale_values ---
def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.to_float(im) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

# --- from google__uncertainty-baselines::uncertainty_baselines/datasets/augment_utils.py::autocontrast.scale_channel.scale_values ---
def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

# --- from terrytangyuan__distributed-ml-patterns::code/project/code/multi-worker-distributed-training.py::make_datasets_unbatched ---
def make_datasets_unbatched():
  BUFFER_SIZE = 10000

  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label
  # Use Fashion-MNIST: https://www.tensorflow.org/datasets/catalog/fashion_mnist
  datasets, _ = tfds.load(name='fashion_mnist', with_info=True, as_supervised=True)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

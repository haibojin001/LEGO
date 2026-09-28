# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg665::PIL.Image.fromarray+numpy.array+numpy.copyto
# name: PIL_numpy_primitive
# summary: Uses PIL.Image.fromarray, numpy.array, numpy.copyto, numpy.uint8 across 2 repos
# anchor_symbols: ['PIL.Image.fromarray', 'numpy.array', 'numpy.copyto', 'numpy.uint8']
# observed in 2 repos: ['ahmetozlu__tensorflow_object_counting_api', 'ahmetozlu__vehicle_counting_tensorflow']...

# --- from ahmetozlu__tensorflow_object_counting_api::utils/visualization_utils.py::save_image_array_as_png ---
def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')

# --- from ahmetozlu__vehicle_counting_tensorflow::utils/visualization_utils.py::save_image_array_as_png ---
def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')

# --- from ahmetozlu__tensorflow_object_counting_api::utils/visualization_utils.py::encode_image_array_as_png_str ---
def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string

# --- from ahmetozlu__vehicle_counting_tensorflow::utils/visualization_utils.py::encode_image_array_as_png_str ---
def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string

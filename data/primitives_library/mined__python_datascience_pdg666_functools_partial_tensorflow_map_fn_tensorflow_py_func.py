# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg666::functools.partial+tensorflow.map_fn+tensorflow.py_func
# name: functools_tensorflow_primitive
# summary: Uses functools.partial, tensorflow.map_fn, tensorflow.py_func across 2 repos
# anchor_symbols: ['functools.partial', 'tensorflow.map_fn', 'tensorflow.py_func']
# observed in 2 repos: ['ahmetozlu__tensorflow_object_counting_api', 'ahmetozlu__vehicle_counting_tensorflow']...

# --- from ahmetozlu__tensorflow_object_counting_api::utils/visualization_utils.py::draw_bounding_boxes_on_image_tensors ---
def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2):
  """Draws bounding boxes on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C].
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  visualize_boxes_fn = functools.partial(
      visualize_boxes_and_labels_on_image_array,
      category_index=category_index,
      instance_masks=None,
      keypoints=None,
      use_normalized_coordinates=True,
      max_boxes_to_draw=max_boxes_to_draw,
      min_score_thresh=min_score_thresh,
      agnostic_mode=False,
      line_thickness=4)

  def draw_boxes(image_boxes_classes_scores):
    """Draws boxes on image."""
    (image, boxes, classes, scores) = image_boxes_classes_scores
    image_with_boxes = tf.py_func(visualize_boxes_fn,
                                  [image, boxes, classes, scores], tf.uint8)
    return image_with_boxes

  images = tf.map_fn(
      draw_boxes, (images, boxes, classes, scores),
      dtype=tf.uint8,
      back_prop=False)
  return images

# --- from ahmetozlu__vehicle_counting_tensorflow::utils/visualization_utils.py::draw_bounding_boxes_on_image_tensors ---
def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2):
  """Draws bounding boxes on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C].
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  visualize_boxes_fn = functools.partial(
      visualize_boxes_and_labels_on_image_array,
      category_index=category_index,
      instance_masks=None,
      keypoints=None,
      use_normalized_coordinates=True,
      max_boxes_to_draw=max_boxes_to_draw,
      min_score_thresh=min_score_thresh,
      agnostic_mode=False,
      line_thickness=4)

  def draw_boxes(image_boxes_classes_scores):
    """Draws boxes on image."""
    (image, boxes, classes, scores) = image_boxes_classes_scores
    image_with_boxes = tf.py_func(visualize_boxes_fn,
                                  [image, boxes, classes, scores], tf.uint8)
    return image_with_boxes

  images = tf.map_fn(
      draw_boxes, (images, boxes, classes, scores),
      dtype=tf.uint8,
      back_prop=False)
  return images

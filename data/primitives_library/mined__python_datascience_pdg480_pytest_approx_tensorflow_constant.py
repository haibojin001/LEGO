# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg480::pytest.approx+tensorflow.constant
# name: pytest_tensorflow_primitive
# summary: Uses pytest.approx, tensorflow.constant across 2 repos
# anchor_symbols: ['pytest.approx', 'tensorflow.constant']
# observed in 2 repos: ['ahmetozlu__tensorflow_object_counting_api', 'stellargraph__stellargraph']...

# --- from stellargraph__stellargraph::tests/layer/test_link_inference.py::make_orthonormal_vectors ---
def make_orthonormal_vectors(dim):
    x_src = np.random.randn(dim)
    x_src /= np.linalg.norm(x_src)  # normalize x_src
    x_dst = np.random.randn(dim)
    x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
    x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

    # Check the IP is zero for numpy operations
    assert np.dot(x_src, x_dst) == pytest.approx(0)

    return x_src, x_dst

# --- from stellargraph__stellargraph::stellargraph/layer/link_inference.py::link_inference.edge_function ---
def edge_function(x):
        le = LinkEmbedding(activation="linear", method=edge_embedding_method)(x)

        # All methods apart from inner product have a dense layer
        # to convert link embedding to the desired output
        if edge_embedding_method in ["ip", "dot"]:
            out = Activation(output_act)(le)
        else:
            out = Dense(output_dim, activation=output_act)(le)

        # Reshape outputs
        out = Reshape((output_dim,))(out)

        if clip_limits:
            out = LeakyClippedLinear(
                low=clip_limits[0], high=clip_limits[1], alpha=0.1
            )(out)
        return out

# --- from ahmetozlu__tensorflow_object_counting_api::smurf_counter_training/legacy/trainer_test.py::FakeDetectionModel.loss ---
def loss(self, prediction_dict, true_image_shapes):
    """Compute scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding predicted tensors
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      a dictionary mapping strings (loss names) to scalar tensors representing
        loss values.
    """
    batch_reg_targets = tf.stack(
        self.groundtruth_lists(fields.BoxListFields.boxes))
    batch_cls_targets = tf.stack(
        self.groundtruth_lists(fields.BoxListFields.classes))
    weights = tf.constant(
        1.0, dtype=tf.float32,
        shape=[len(self.groundtruth_lists(fields.BoxListFields.boxes)), 1])

    location_losses = self._localization_loss(
        prediction_dict['box_encodings'], batch_reg_targets,
        weights=weights)
    cls_losses = self._classification_loss(
        prediction_dict['class_predictions_with_background'], batch_cls_targets,
        weights=weights)

    loss_dict = {
        'localization_loss': tf.reduce_sum(location_losses),
        'classification_loss': tf.reduce_sum(cls_losses),
    }
    return loss_dict

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg653::numpy.expand_dims+numpy.squeeze
# name: numpy_primitive
# summary: Uses numpy.expand_dims, numpy.squeeze across 2 repos
# anchor_symbols: ['numpy.expand_dims', 'numpy.squeeze']
# observed in 2 repos: ['ahmetozlu__tensorflow_object_counting_api', 'zama-ai__concrete-ml']...

# --- from ahmetozlu__tensorflow_object_counting_api::detection_layer.py::ObjectDetector.get_localization ---
def get_localization(self, image, visual=False):         
        category_index={1: {'id': 1, 'name': u'player'}}  
        
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})          
              if visual == True:
                  visualization_utils.visualize_boxes_and_labels_on_image_array_tracker(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.4,
                      line_thickness=3)   
                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()               
              boxes=np.squeeze(boxes)
              classes =np.squeeze(classes)
              scores = np.squeeze(scores)  
              cls = classes.tolist()
              idx_vec = [i for i, v in enumerate(cls) if ((scores[i]>0.6))]              
              if len(idx_vec) ==0:
                  print('there are not any detections, passing to the next frame...')
              else:
                  tmp_object_boxes=[]
                  for idx in idx_vec:
                      dim = image.shape[0:2]
                      box = self.box_normal_to_pixel(boxes[idx], dim)
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                      ratio = box_h/(box_w + 0.01)
                      
                      #if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                      tmp_object_boxes.append(box)
                      #print(box, ', confidence: ', scores[idx], 'ratio:', ratio)                                                   
                  
                  self.object_boxes = tmp_object_boxes             
        return self.object_boxes

# --- from zama-ai__concrete-ml::src/concrete/ml/onnx/ops_impl.py::numpy_conv ---
def numpy_conv(
    x: numpy.ndarray,
    w: numpy.ndarray,
    b: Optional[numpy.ndarray] = None,
    *,
    dilations: Tuple[int, ...],
    group: int = 1,
    kernel_shape: Tuple[int, ...],
    pads: Tuple[int, ...],
    strides: Tuple[int, ...],
) -> Tuple[numpy.ndarray]:
    """Compute N-D convolution using Torch.

    Currently supports 2d convolution with torch semantics. This function is also ONNX compatible.

    See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv

    Args:
        x (numpy.ndarray): input data (many dtypes are supported). Shape is N x C x H x W for 2d
        w (numpy.ndarray): weights tensor. Shape is (O x I x Kh x Kw) for 2d
        b (Optional[numpy.ndarray]): bias tensor, Shape is (O,). Default to None.
        dilations (Tuple[int, ...]): dilation of the kernel, default 1 on all dimensions.
        group (int): number of convolution groups, can be 1 or a multiple of both (C,) and (O,), so
            that I = C / group. Default to 1.
        kernel_shape (Tuple[int, ...]): shape of the kernel. Should have 2 elements for 2d conv
        pads (Tuple[int, ...]): padding in ONNX format (begin, end) on each axis
        strides (Tuple[int, ...]): stride of the convolution on each axis

    Returns:
        res (numpy.ndarray): a tensor of size (N x OutChannels x OutHeight x OutWidth).
           See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    """

    # Convert the inputs to tensors to compute conv using torch
    assert_true(
        len(kernel_shape) in (1, 2),
        f"The convolution operator currently only supports 1d or 2d. Got {len(kernel_shape)}-d",
    )
    assert_true(
        bool(numpy.all(numpy.asarray(dilations) == 1)),
        "The convolution operator in Concrete does not support dilation",
    )

    weight_channels = x.shape[1]
    assert_true(
        w.shape[1] == weight_channels / group,
        f"Expected number of channels in weight to be {weight_channels / group} (C / group). Got "
        f"{w.shape[1]}.",
    )

    assert_true(
        w.shape[0] % group == 0,
        f"Expected number of output O ({w.shape[0]}) to be a multiple of group " f"({group}).",
    )

    # Pad the input if needed
    x_pad = numpy_onnx_pad(x, pads)

    is_conv1d = len(kernel_shape) == 1

    # Workaround for handling torch's Conv1d operator until it is supported by Concrete Python
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4117
    if is_conv1d:
        x_pad = numpy.expand_dims(x_pad, axis=-2)
        w = numpy.expand_dims(w, axis=-2)
        kernel_shape = (1, kernel_shape[0])
        strides = (1, strides[0])
        dilations = (1, dilations[0])

    # Compute the torch convolution
    res = fhe_conv(
        x=x_pad,
        weight=w,
        bias=b,
        pads=None,
        strides=strides,
        dilations=dilations,
        kernel_shape=kernel_shape,
        group=group,
    )

    # Workaround for handling torch's Conv1d operator until it is supported by Concrete Python
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4117
    if is_conv1d:
        res = numpy.squeeze(res, axis=-2)

    return (res,)

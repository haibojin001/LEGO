# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg33::functools.partial+inspect.getargspec+inspect.getfullargspec
# name: functools_inspect_primitive
# summary: Uses functools.partial, inspect.getargspec, inspect.getfullargspec, inspect.signature across 7 repos
# anchor_symbols: ['functools.partial', 'inspect.getargspec', 'inspect.getfullargspec', 'inspect.signature', 'keras.backend.abs', 'keras.backend.backend']
# observed in 7 repos: ['DeepWisdom__AutoDL', 'HDI-Project__ATM', 'ahmetozlu__tensorflow_object_counting_api', 'd2l-ai__d2l-en', 'donnemartin__data-science-ipython-notebooks']...

# --- from keras-team__keras-contrib::tests/keras_contrib/utils/save_load_utils_test.py::test_save_and_load_all_weights.make_model ---
def make_model():
        _x = Input((10,))
        _y = Dense(10)(_x)
        _m = Model(_x, _y)
        _m.compile('adam', 'mean_squared_error')
        _m._make_train_function()
        return _m

# --- from HDI-Project__ATM::atm/compat.py::getargs ---
def getargs(function):
    """Get the function arguments using inspect."""
    if PY2:
        return inspect.getargspec(function).args
    else:
        return inspect.getfullargspec(function).args

# --- from keras-team__keras-contrib::keras_contrib/applications/resnet.py::_bn_relu ---
def _bn_relu(x, bn_name=None, relu_name=None):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)

# --- from lazyprogrammer__machine_learning_examples::nlp_class3/simple_rnn_test.py::lstm1 ---
def lstm1():
  input_ = Input(shape=(T, D))
  rnn = LSTM(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h, c = model.predict(X)
  print("o:", o)
  print("h:", h)
  print("c:", c)

# --- from lazyprogrammer__machine_learning_examples::nlp_class3/simple_rnn_test.py::lstm2 ---
def lstm2():
  input_ = Input(shape=(T, D))
  rnn = LSTM(M, return_state=True, return_sequences=True)
  # rnn = GRU(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h, c = model.predict(X)
  print("o:", o)
  print("h:", h)
  print("c:", c)

# --- from ahmetozlu__tensorflow_object_counting_api::mask_rcnn_counting_api/spaghetti_counter_training/training/mrcnn/model.py::smooth_l1_loss ---
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

# --- from ahmetozlu__tensorflow_object_counting_api::mask_rcnn_counting_api/spaghetti_counter_training/mrcnn/model.py::smooth_l1_loss ---
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

# --- from d2l-ai__d2l-en::d2l/tensorflow.py::FashionMNIST.get_dataloader ---
def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        data = self.train if train else self.val
        process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                                tf.cast(y, dtype='int32'))
        resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
        shuffle_buf = len(data[0]) if train else 1
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)

# --- from d2l-ai__d2l-en::d2l/jax.py::FashionMNIST.get_dataloader ---
def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        data = self.train if train else self.val
        process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                                tf.cast(y, dtype='int32'))
        resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
        shuffle_buf = len(data[0]) if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))

# --- from donnemartin__data-science-ipython-notebooks::deep-learning/keras-tutorial/deep_learning_models/resnet50.py::identity_block ---
def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/at_speech/backbones/tr34_bb.py::resnet_2D_v1 ---
def resnet_2D_v1(input_dim, mode='train'):
    bn_axis = 3
    if mode == 'train':
        inputs = Input(shape=input_dim, name='input')
    else:
        inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
    x1 = Conv2D(64, (7, 7),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1/3x3_s1')(inputs)
    x1 = BatchNormalization(axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)
    x2 = conv_block_2D(x1, 3, [48, 48, 96], stage=2, block='a', strides=(1, 1), trainable=True)
    x2 = identity_block_2D(x2, 3, [48, 48, 96], stage=2, block='b', trainable=True)
    x3 = conv_block_2D(x2, 3, [96, 96, 128], stage=3, block='a', trainable=True)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='b', trainable=True)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='c', trainable=True)
    x4 = conv_block_2D(x3, 3, [128, 128, 256], stage=4, block='a', trainable=True)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='b', trainable=True)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='c', trainable=True)
    x5 = conv_block_2D(x4, 3, [256, 256, 512], stage=5, block='a', trainable=True)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='b', trainable=True)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='c', trainable=True)
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/at_speech/backbones/tr34_bb.py::identity_block_2D ---
def identity_block_2D(input_tensor, kernel_size, filters, stage, block, trainable=True):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
    x = Activation('relu')(x)
    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
    x = Activation('relu')(x)
    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_3)(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

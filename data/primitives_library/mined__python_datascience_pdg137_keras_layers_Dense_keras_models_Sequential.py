# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg137::keras.layers.Dense+keras.models.Sequential
# name: keras_primitive
# summary: Uses keras.layers.Dense, keras.models.Sequential across 5 repos
# anchor_symbols: ['keras.layers.Dense', 'keras.models.Sequential']
# observed in 5 repos: ['ClimbsRocks__auto_ml', 'HunterMcGushion__hyperparameter_hunter', 'keras-team__keras-contrib', 'microsoft__nni', 'ottogroup__palladium']...

# --- from keras-team__keras-contrib::tests/keras_contrib/callbacks/cyclical_learning_rate_test.py::build_model ---
def build_model():
    model = Sequential([
        Dense(2, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    return model

# --- from keras-team__keras-contrib::keras_contrib/tests/optimizers.py::get_model ---
def get_model(input_dim, num_hidden, output_dim):
    model = Sequential()
    model.add(Dense(num_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model

# --- from ottogroup__palladium::examples/keras/model.py::keras_model ---
def keras_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
        )
    return model

# --- from ClimbsRocks__auto_ml::auto_ml/utils_models.py::get_activation_layer ---
def get_activation_layer(activation):
    if activation == 'LeakyReLU':
        return LeakyReLU()
    if activation == 'PReLU':
        return PReLU()
    if activation == 'ELU':
        return ELU()
    if activation == 'ThresholdedReLU':
        return ThresholdedReLU()

    return Activation(activation)

# --- from HunterMcGushion__hyperparameter_hunter::tests/smoke_tests/test_keras.py::_build_fn_orthogonal_i_0 ---
def _build_fn_orthogonal_i_0(input_shape):  # `"orthogonal"`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer="orthogonal"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- from HunterMcGushion__hyperparameter_hunter::tests/smoke_tests/test_keras.py::_build_fn_orthogonal_i_1 ---
def _build_fn_orthogonal_i_1(input_shape):  # `orthogonal()`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=orthogonal()),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- from microsoft__nni::examples/trials/mnist-batch-tune-keras/mnist-keras.py::create_mnist_model ---
def create_mnist_model(hyper_params, input_shape=(H, W, 1), num_classes=NUM_CLASSES):
    '''
    Create simple convolutional model
    '''
    layers = [
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ]

    model = Sequential(layers)

    if hyper_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(lr=hyper_params['learning_rate'], momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

# --- from microsoft__nni::examples/trials/mnist-keras/mnist-keras.py::create_mnist_model ---
def create_mnist_model(hyper_params, input_shape=(H, W, 1), num_classes=NUM_CLASSES):
    '''
    Create simple convolutional model
    '''
    layers = [
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ]

    model = Sequential(layers)

    if hyper_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(lr=hyper_params['learning_rate'], momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

# --- from ClimbsRocks__auto_ml::auto_ml/utils_models.py::make_deep_learning_classifier ---
def make_deep_learning_classifier(hidden_layers=None, num_cols=None, optimizer='Adadelta', dropout_rate=0.2, weight_constraint=0, final_activation='sigmoid', feature_learning=False, activation='elu', kernel_initializer='normal'):

    if feature_learning == True and hidden_layers is None:
        hidden_layers = [1, 0.75, 0.25]

    if hidden_layers is None:
        hidden_layers = [1, 0.75, 0.25]

    # The hidden_layers passed to us is simply describing a shape. it does not know the num_cols we are dealing with, it is simply values of 0.5, 1, and 2, which need to be multiplied by the num_cols
    scaled_layers = []
    for layer in hidden_layers:
        scaled_layers.append(min(int(num_cols * layer), 10))

    # If we're training this model for feature_learning, our penultimate layer (our final hidden layer before the "output" layer) will always have 10 neurons, meaning that we always output 10 features from our feature_learning model
    if feature_learning == True:
        scaled_layers.append(10)


    model = Sequential()

    # There are times we will want the output from our penultimate layer, not the final layer, so give it a name that makes the penultimate layer easy to find
    model.add(Dense(scaled_layers[0], input_dim=num_cols, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
    model.add(get_activation_layer(activation))

    for layer_size in scaled_layers[1:-1]:
        model.add(Dense(layer_size, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
        model.add(get_activation_layer(activation))

    model.add(Dense(scaled_layers[-1], kernel_initializer=kernel_initializer, name='penultimate_layer', kernel_regularizer=regularizers.l2(0.01)))
    model.add(get_activation_layer(activation))

    model.add(Dense(1, kernel_initializer=kernel_initializer, activation=final_activation))
    model.compile(loss='binary_crossentropy', optimizer=get_optimizer(optimizer), metrics=['accuracy', 'poisson'])
    return model

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg293::mmoe.MMoE+tensorflow.keras.initializers.VarianceScaling+tensorflow.keras.layers.Dense
# name: mmoe_tensorflow_primitive
# summary: Uses mmoe.MMoE, tensorflow.keras.initializers.VarianceScaling, tensorflow.keras.layers.Dense, tensorflow.keras.layers.Input across 2 repos
# anchor_symbols: ['mmoe.MMoE', 'tensorflow.keras.initializers.VarianceScaling', 'tensorflow.keras.layers.Dense', 'tensorflow.keras.layers.Input', 'tensorflow.keras.models.Model', 'tensorflow.keras.optimizers.Adam']
# observed in 2 repos: ['drawbridge__keras-mmoe', 'lazyprogrammer__machine_learning_examples']...

# --- from lazyprogrammer__machine_learning_examples::tf2.0/rl_trader.py::mlp ---
def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
  """ A multi-layer perceptron """

  # input layer
  i = Input(shape=(input_dim,))
  x = i

  # hidden layers
  for _ in range(n_hidden_layers):
    x = Dense(hidden_dim, activation='relu')(x)
  
  # final layer
  x = Dense(n_action)(x)

  # make the model
  model = Model(i, x)

  model.compile(loss='mse', optimizer='adam')
  print((model.summary()))
  return model

# --- from drawbridge__keras-mmoe::synthetic_demo.py::main ---
def main():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label = data_preparation()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    # Set up the input layer
    input_layer = Input(shape=(num_features,))

    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=16,
        num_experts=8,
        num_tasks=2
    )(input_layer)

    output_layers = []

    output_info = ['y0', 'y1']

    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=1,
            name=output_info[index],
            activation='linear',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    learning_rates = [1e-4, 1e-3, 1e-2]
    adam_optimizer = Adam(lr=learning_rates[0])
    model.compile(
        loss={'y0': 'mean_squared_error', 'y1': 'mean_squared_error'},
        optimizer=adam_optimizer,
        metrics=[metrics.mae]
    )

    # Print out model architecture summary
    model.summary()

    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        epochs=100
    )

# --- from drawbridge__keras-mmoe::census_income_demo.py::main ---
def main():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    # Set up the input layer
    input_layer = Input(shape=(num_features,))

    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=4,
        num_experts=8,
        num_tasks=2
    )(input_layer)

    output_layers = []

    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=output_info[index][0],
            name=output_info[index][1],
            activation='softmax',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    adam_optimizer = Adam()
    model.compile(
        loss={'income': 'binary_crossentropy', 'marital': 'binary_crossentropy'},
        optimizer=adam_optimizer,
        metrics=['accuracy']
    )

    # Print out model architecture summary
    model.summary()

    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ],
        epochs=100
    )

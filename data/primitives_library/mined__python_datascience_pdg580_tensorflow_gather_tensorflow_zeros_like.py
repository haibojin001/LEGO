# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg580::tensorflow.gather+tensorflow.zeros_like
# name: tensorflow_primitive
# summary: Uses tensorflow.gather, tensorflow.zeros_like across 2 repos
# anchor_symbols: ['tensorflow.gather', 'tensorflow.zeros_like']
# observed in 2 repos: ['google__uncertainty-baselines', 'pykale__pykale']...

# --- from google__uncertainty-baselines::baselines/jft/data_uncertainty_utils.py::create_cifar10_to_cifar10h_fn.convert ---
def convert(example):
    idx = idx_map.lookup(example['id'])
    if idx == -1:
      logging.warn('Index -1 encountered in the CIFAR-10H dataset.')
      example['labels'] = tf.zeros_like(tf.gather(cifar10h_probs, 0))
      example['count'] = tf.zeros_like(tf.gather(cifar10h_counts, 0))
    else:
      example['labels'] = tf.gather(cifar10h_probs, idx)
      example['count'] = tf.gather(cifar10h_counts, idx)
    return example

# --- from google__uncertainty-baselines::baselines/jft/data_uncertainty_utils.py::create_imagenet_to_real_fn.convert ---
def convert(example: Features) -> Features:
    idx = idx_map.lookup(example['file_name'])
    if idx == -1:
      logging.warn('Index -1 encountered in the ImageNet real dataset.')
      example['labels'] = tf.zeros_like(tf.gather(real_probs, 0))
      example['mask'] = tf.zeros_like(tf.gather(real_weights, 0))
    else:
      example['labels'] = tf.gather(real_probs, idx)
      example['mask'] = tf.gather(real_weights, idx)
    return example

# --- from pykale__pykale::kale/loaddata/avmnist_datasets.py::AVMNISTDataset.load_data ---
def load_data(self):
        trains = [
            np.load(self.data_dir + "/image/train_data.npy"),
            np.load(self.data_dir + "/audio/train_data.npy"),
            np.load(self.data_dir + "/train_labels.npy"),
        ]
        tests = [
            np.load(self.data_dir + "/image/test_data.npy"),
            np.load(self.data_dir + "/audio/test_data.npy"),
            np.load(self.data_dir + "/test_labels.npy"),
        ]
        train_valid_size = len(trains[0])
        test_size = len(tests[0])

        if self.flatten_audio:
            trains[1] = trains[0].reshape(train_valid_size, 112 * 112)
            tests[1] = tests[0].reshape(test_size, 112 * 112)

        if self.normalize_image:
            trains[0] = trains[0].astype("float64")
            trains[0] /= 255.0
            tests[0] = tests[0].astype("float64")
            tests[0] /= 255.0
        if self.normalize_audio:
            trains[1] = trains[1].astype("float64")
            trains[1] /= 255.0
            tests[1] = tests[1].astype("float64")
            tests[1] /= 255.0
        if not self.flatten_image:
            trains[0] = trains[0].reshape(train_valid_size, 28, 28)
            tests[0] = tests[0].reshape(test_size, 28, 28)
        if self.unsqueeze_channel:
            trains[0] = np.expand_dims(trains[0], 1)
            tests[0] = np.expand_dims(tests[0], 1)
            trains[1] = np.expand_dims(trains[1], 1)
            tests[1] = np.expand_dims(tests[1], 1)
        trains[2] = trains[2].astype(int)
        tests[2] = tests[2].astype(int)

        self.train_valid_data = [[trains[j][i] for j in range(3)] for i in range(train_valid_size)]
        self.test_data = [[tests[j][i] for j in range(3)] for i in range(test_size)]

        train_size = int(train_valid_size * 0.9)
        valid_size = train_valid_size - train_size

        self.train_data, self.valid_data = torch.utils.data.random_split(
            self.train_valid_data, [train_size, valid_size]
        )

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg154::tensorflow.compat.v1.exp+tensorflow.compat.v1.log+tensorflow.compat.v1.reduce_max
# name: tensorflow_primitive
# summary: Uses tensorflow.compat.v1.exp, tensorflow.compat.v1.log, tensorflow.compat.v1.reduce_max, tensorflow.compat.v1.reduce_sum across 2 repos
# anchor_symbols: ['tensorflow.compat.v1.exp', 'tensorflow.compat.v1.log', 'tensorflow.compat.v1.reduce_max', 'tensorflow.compat.v1.reduce_sum']
# observed in 2 repos: ['microsoft__nni', 'tflearn__tflearn']...

# --- from microsoft__nni::nni/algorithms/hpo/ppo_tuner/distri.py::CategoricalPd.entropy ---
def entropy(self):
        """compute entropy"""
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

# --- from microsoft__nni::nni/algorithms/hpo/ppo_tuner/distri.py::CategoricalPd.kl ---
def kl(self, other):
        """kl"""
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

# --- from tflearn__tflearn::examples/images/variational_autoencoder.py::vae_loss ---
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

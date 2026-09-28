# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg299::keras.backend.exp+keras.backend.max+keras.backend.sum
# name: keras_primitive
# summary: Uses keras.backend.exp, keras.backend.max, keras.backend.sum across 2 repos
# anchor_symbols: ['keras.backend.exp', 'keras.backend.max', 'keras.backend.sum']
# observed in 2 repos: ['DeepWisdom__AutoDL', 'lazyprogrammer__machine_learning_examples']...

# --- from lazyprogrammer__machine_learning_examples::nlp_class3/attention.py::softmax_over_time ---
def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/at_speech/backbones/thinresnet34.py::VladPooling.call ---
def call(self, x):
        feat, cluster_score = x
        num_features = feat.shape[-1]
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims = True)
        A = K.expand_dims(A, -1)
        feat_broadcast = K.expand_dims(feat, -2)
        feat_res = feat_broadcast - self.cluster
        weighted_res = tf.multiply(A, feat_res)
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs

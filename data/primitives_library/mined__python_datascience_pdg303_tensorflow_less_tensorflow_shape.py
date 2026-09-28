# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg303::tensorflow.less+tensorflow.shape
# name: tensorflow_primitive
# summary: Uses tensorflow.less, tensorflow.shape across 2 repos
# anchor_symbols: ['tensorflow.less', 'tensorflow.shape']
# observed in 2 repos: ['google__uncertainty-baselines', 'lazyprogrammer__machine_learning_examples']...

# --- from lazyprogrammer__machine_learning_examples::nlp_class2/rntn_tensorflow_rnn.py::RecursiveNN.fit.condition ---
def condition(hiddens, n):
            # loop should continue while n < len(words)
            return tf.less(n, tf.shape(input=words)[0])

# --- from google__uncertainty-baselines::baselines/drug_cardiotoxicity/augmentation_utils.py::GraphAugment.subgraph._subgraph_helper._condition_continue_random_walk ---
def _condition_continue_random_walk(num_nodes_seen, idx_neighbors,
                                          idx_nodes_in_subgraph,
                                          num_nodes_in_subgraph,
                                          total_num_nodes):
        del idx_neighbors  # Unused argument
        subgraph_max_size_not_reached = tf.less(
            tf.shape(idx_nodes_in_subgraph)[0], num_nodes_in_subgraph)
        valid_nodes_left = tf.math.less(num_nodes_seen, total_num_nodes)
        return tf.logical_and(subgraph_max_size_not_reached, valid_nodes_left)

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg38::random.choice+random.random
# name: random_primitive
# summary: Uses random.choice, random.random across 6 repos
# anchor_symbols: ['random.choice', 'random.random']
# observed in 6 repos: ['DeepWisdom__AutoDL', 'WecoAI__aideml', 'capitalone__DataProfiler', 'd2l-ai__d2l-en', 'google__uncertainty-baselines']...

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Video/skeleton/data/augmentations.py::Augmentation.__call__ ---
def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Image/skeleton/data/augmentations.py::Augmentation.__call__ ---
def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

# --- from d2l-ai__d2l-en::d2l/mxnet.py::_get_next_sentence ---
def _get_next_sentence(sentence, next_sentence, paragraphs):
    """Defined in :numref:`sec_bert-dataset`"""
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

# --- from d2l-ai__d2l-en::d2l/torch.py::_get_next_sentence ---
def _get_next_sentence(sentence, next_sentence, paragraphs):
    """Defined in :numref:`sec_bert-dataset`"""
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

# --- from lazyprogrammer__machine_learning_examples::nlp_class/article_spinner.py::test_spinner ---
def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("Original:", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2: # 20% chance of replacement
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print("Spun:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))

# --- from capitalone__DataProfiler::dataprofiler/tests/profilers/utils.py::generate_random_date_sample ---
def generate_random_date_sample(
    start_date, end_date, date_formats=["%Y-%m-%d %H:%M:%S"], t=None
):
    """
    Generate a synthetic date
    :param start_date: earliest date in date
    :type start_date: pandas.tslib.Timestamp
    :param end_date: latest date in date
    :type end_date: pandas.tslib.Timestamp
    :param date_formats: possible formats for date
    :type date_formats: list[str]
    :return: sythetic date
    :rtype: str
    """
    if not t:
        t = random.random()

    random_format = random.choice(date_formats)
    ptime = start_date + t * (end_date - start_date)

    return ptime.strftime(random_format)

# --- from google__uncertainty-baselines::baselines/drug_cardiotoxicity/augmentation_utils.py::GraphAugment.augment ---
def augment(self, input_graph: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Randomly selects and performs an augmentation from all possible.

    Args:
      input_graph: Graph features to be augmented.

    Returns:
      Augmented graph.
    """
    # TODO(jihyeonlee): Allow user to specify number of augmentations to perform
    # per graph. Consider allowing different aug_ratio by function.
    if self.aug_ratio == 0. or not self.augmentations_to_use or random.random(
    ) < self.aug_prob or self.aug_prob == 0.:
      return input_graph
    aug_function_name = random.choice(self.augmentations_to_use)
    output_graph = input_graph.copy()
    augmented_graph, _ = self.augmentations_available[aug_function_name](
        input_graph)
    output_graph.update(augmented_graph)
    return output_graph

# --- from WecoAI__aideml::aide/agent.py::Agent.search_policy ---
def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] greedy node selected")
        return greedy_node

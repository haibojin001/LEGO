# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg234::torch.any+torch.isnan
# name: torch_primitive
# summary: Uses torch.any, torch.isnan across 2 repos
# anchor_symbols: ['torch.any', 'torch.isnan']
# observed in 2 repos: ['Lightning-AI__torchmetrics', 'allenai__allennlp']...

# --- from Lightning-AI__torchmetrics::src/torchmetrics/wrappers/multioutput.py::_get_nan_indices ---
def _get_nan_indices(*tensors: Tensor) -> Tensor:
    """Get indices of rows along dim 0 which have NaN values."""
    if len(tensors) == 0:
        raise ValueError("Must pass at least one tensor as argument")
    sentinel = tensors[0]
    nan_idxs = torch.zeros(len(sentinel), dtype=torch.bool, device=sentinel.device)
    for tensor in tensors:
        permuted_tensor = tensor.flatten(start_dim=1)
        nan_idxs |= torch.any(torch.isnan(permuted_tensor), dim=1)
    return nan_idxs

# --- from allenai__allennlp::tests/modules/token_embedders/pretrained_transformer_mismatched_embedder_test.py::TestPretrainedTransformerMismatchedEmbedder.test_exotic_tokens_no_nan_grads ---
def test_exotic_tokens_no_nan_grads(self):
        token_indexer = PretrainedTransformerMismatchedIndexer("bert-base-uncased")

        sentence1 = ["A", "", "AllenNLP", "sentence", "."]
        sentence2 = ["A", "\uf732\uf730\uf730\uf733", "AllenNLP", "sentence", "."]

        tokens1 = [Token(word) for word in sentence1]
        tokens2 = [Token(word) for word in sentence2]
        vocab = Vocabulary()

        token_embedder = BasicTextFieldEmbedder(
            {"bert": PretrainedTransformerMismatchedEmbedder("bert-base-uncased")}
        )

        instance1 = Instance({"tokens": TextField(tokens1, {"bert": token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens2, {"bert": token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        bert_vectors = token_embedder(tokens)
        test_loss = bert_vectors.mean()

        test_loss.backward()

        for name, param in token_embedder.named_parameters():
            grad = param.grad
            assert (grad is None) or (not torch.any(torch.isnan(grad)).item())

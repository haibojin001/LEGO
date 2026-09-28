# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg140::numpy.allclose+numpy.array+numpy.expand_dims
# name: numpy_torch_primitive
# summary: Uses numpy.allclose, numpy.array, numpy.expand_dims, torch.from_numpy across 3 repos
# anchor_symbols: ['numpy.allclose', 'numpy.array', 'numpy.expand_dims', 'torch.from_numpy']
# observed in 3 repos: ['allenai__allennlp', 'microsoft__nni', 'zama-ai__concrete-ml']...

# --- from microsoft__nni::examples/trials/kaggle-tgs-salt/loader.py::mask_to_tensor ---
def mask_to_tensor(x):
    x = np.array(x).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    return x

# --- from allenai__allennlp::tests/modules/masked_layer_norm_test.py::TestMaskedLayerNorm.test_masked_layer_norm ---
def test_masked_layer_norm(self):
        x_n = np.random.rand(2, 3, 7)
        mask_n = np.array([[1, 1, 0], [1, 1, 1]])

        x = torch.from_numpy(x_n).float()
        mask = torch.from_numpy(mask_n).bool()

        layer_norm = MaskedLayerNorm(7, gamma0=0.2)
        normed_x = layer_norm(x, mask)

        N = 7 * 5
        mean = (x_n * np.expand_dims(mask_n, axis=-1)).sum() / N
        std = np.sqrt(
            (((x_n - mean) * np.expand_dims(mask_n, axis=-1)) ** 2).sum() / N
            + util.tiny_value_of_dtype(torch.float)
        )
        expected = 0.2 * (x_n - mean) / (std + util.tiny_value_of_dtype(torch.float))

        assert np.allclose(normed_x.data.numpy(), expected)

# --- from allenai__allennlp::tests/modules/elmo_test.py::TestElmoTokenRepresentation.test_elmo_token_representation_bos_eos ---
def test_elmo_token_representation_bos_eos(self):
        # The additional <S> and </S> embeddings added by the embedder should be as expected.
        indexer = ELMoTokenCharactersIndexer()

        elmo_token_embedder = _ElmoCharacterEncoder(self.options_file, self.weight_file)

        for correct_index, token in [[0, "<S>"], [2, "</S>"]]:
            indices = indexer.tokens_to_indices([Token(token)], Vocabulary())
            indices = torch.from_numpy(numpy.array(indices["elmo_tokens"])).view(1, 1, -1)
            embeddings = elmo_token_embedder(indices)["token_embedding"]
            assert numpy.allclose(
                embeddings[0, correct_index, :].data.numpy(), embeddings[0, 1, :].data.numpy()
            )

# --- from zama-ai__concrete-ml::use_case_examples/llm/qgpt2_class.py::QuantizedModel.run_torch ---
def run_torch(self, inputs: torch.Tensor, fhe: str = "disable", true_float: bool = False):
        """Run the quantized operators, with additional pre and post-processing steps.

        This method is used to take and output torch tensors with floating points.

        Args:
            inputs (torch.Tensor): The input values to consider, in floating points.
            fhe (str): The FHE mode to consider, either "disable", "simulate" or "execute". Default
                to "disable".
            true_float (bool): If the FHE mode is set to "disable", indicate if the operations
                should be in floating points instead of being quantized. Default to False.

        Returns:
            torch.Tensor: The output values, in floating points.
        """

        # Convert the torch tensor to a numpy array
        inputs = inputs.detach().cpu().numpy()

        # Store the inputs as the calibration values. This is done in order to be able to easily
        # compile the model without having to manually extract the model's intermediary hidden
        # states. More importantly, these values are used to convert the quantized inputs from the
        # run_numpy method into their DualArray equivalent, as the compiler only accepts Numpy
        # arrays
        self.x_calib = inputs

        # Quantize the inputs
        q_inputs = self.quantizer.quantize(inputs, key="inputs_quant")

        # If the FHE mode is set to disable, we only need to run the quantized operators in the
        # clear and de-quantize
        if fhe == "disable":
            q_y = self.run_numpy(q_inputs)

            if true_float:
                # Directly returning the output DualArray's floating points does not propagate the
                # quantization parameters. Therefore, these values are the result of float-only
                # computations
                y = q_y.float_array

            else:
                # De-quantizing the output DualArray propagates the quantization parameters. These
                # values should represent the expected values from FHE computations as they are the
                # result of quantized-only computations
                y = q_y.dequantize(key="y_dequant").float_array

        # Else, the FHE circuit, built thanks to the compilation step, needs to be called
        else:
            assert (
                self.circuit is not None
            ), "Module is not compiled. Please run `compile` on a representative inputset."

            # Batched operations is not yet handled by Concrete Python and inputs need to be
            # processed one by one
            y_all = []
            for q_x in q_inputs:

                # The circuit is expecting an input with a batch size of 1 in the first axis
                q_x = np.expand_dims(q_x, axis=0)

                if fhe == "simulate":
                    q_y = self.circuit.simulate(q_x)

                elif fhe == "execute":
                    q_y = self.circuit.encrypt_run_decrypt(q_x)

                else:
                    raise ValueError(
                        "Parameter 'fhe' can only be 'disable', 'simulate' or 'execute'"
                    )

                # The quantizer needs to be directly called in order to de-quantize the circuit's
                # output, as they are here stored in a Numpy array instead of a DualArray object
                y_all.append(self.quantizer.dequantize(q_y, key="y_dequant"))

            y = np.concatenate(y_all)

        # Return the values in a torch tensor, in floating points
        return torch.from_numpy(y).type(torch.float32)

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg640::numpy.arange+numpy.setdiff1d+torch.from_numpy
# name: numpy_torch_primitive
# summary: Uses numpy.arange, numpy.setdiff1d, torch.from_numpy across 2 repos
# anchor_symbols: ['numpy.arange', 'numpy.setdiff1d', 'torch.from_numpy']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/ml_algo/torch_based/linear_model.py::TorchBasedLinearEstimator._prepare_data_dense ---
def _prepare_data_dense(self, data: np.ndarray):
        """Prepare dense matrix.

        Split categorical and numeric features.

        Args:
            data: data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        if 0 < len(self.categorical_idx) < data.shape[1]:
            data_cat = torch.from_numpy(data[:, self.categorical_idx].astype(np.int64))
            data = torch.from_numpy(data[:, np.setdiff1d(np.arange(data.shape[1]), self.categorical_idx)])
            return data, data_cat

        elif len(self.categorical_idx) == 0:
            data = torch.from_numpy(data)
            return data, None

        else:
            data_cat = torch.from_numpy(data.astype(np.int64))
            return None, data_cat

# --- from sberbank-ai-lab__LightAutoML::lightautoml/ml_algo/torch_based/linear_model.py::TorchBasedLinearEstimator._prepare_data_dense ---
def _prepare_data_dense(self, data: np.ndarray):
        """Prepare dense matrix.

        Split categorical and numeric features.

        Args:
            data: data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        if 0 < len(self.categorical_idx) < data.shape[1]:
            data_cat = torch.from_numpy(data[:, self.categorical_idx].astype(np.int64))
            data = torch.from_numpy(data[:, np.setdiff1d(np.arange(data.shape[1]), self.categorical_idx)])
            return data, data_cat

        elif len(self.categorical_idx) == 0:
            data = torch.from_numpy(data)
            return data, None

        else:
            data_cat = torch.from_numpy(data.astype(np.int64))
            return None, data_cat

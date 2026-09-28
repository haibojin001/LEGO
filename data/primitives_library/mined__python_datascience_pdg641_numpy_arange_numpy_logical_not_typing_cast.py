# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg641::numpy.arange+numpy.logical_not+typing.cast
# name: numpy_typing_primitive
# summary: Uses numpy.arange, numpy.logical_not, typing.cast across 2 repos
# anchor_symbols: ['numpy.arange', 'numpy.logical_not', 'typing.cast']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/validation/np_iterators.py::FoldsIterator.__next__ ---
def __next__(self) -> Tuple[np.ndarray, NumpyOrSparse, NumpyOrSparse]:
        """Define how to get next object.

        Returns:
            Mask for current fold, train dataset, validation dataset.

        """
        if self._curr_idx == self.n_folds:
            raise StopIteration
        val_idx = self.train.folds == self._curr_idx
        tr_idx = np.logical_not(val_idx)
        idx = np.arange(self.train.shape[0])
        tr_idx, val_idx = idx[tr_idx], idx[val_idx]
        train, valid = self.train[tr_idx], self.train[val_idx]
        self._curr_idx += 1
        return val_idx, cast(NumpyOrSparse, train), cast(NumpyOrSparse, valid)

# --- from sberbank-ai-lab__LightAutoML::lightautoml/validation/np_iterators.py::FoldsIterator.__next__ ---
def __next__(self) -> Tuple[np.ndarray, NumpyOrSparse, NumpyOrSparse]:
        """Define how to get next object.

        Returns:
            Mask for current fold, train dataset, validation dataset.

        """
        if self._curr_idx == self.n_folds:
            raise StopIteration
        val_idx = self.train.folds == self._curr_idx
        tr_idx = np.logical_not(val_idx)
        idx = np.arange(self.train.shape[0])
        tr_idx, val_idx = idx[tr_idx], idx[val_idx]
        train, valid = self.train[tr_idx], self.train[val_idx]
        self._curr_idx += 1
        return val_idx, cast(NumpyOrSparse, train), cast(NumpyOrSparse, valid)

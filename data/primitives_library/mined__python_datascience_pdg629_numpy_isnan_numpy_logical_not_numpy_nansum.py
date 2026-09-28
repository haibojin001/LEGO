# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg629::numpy.isnan+numpy.logical_not+numpy.nansum
# name: numpy_primitive
# summary: Uses numpy.isnan, numpy.logical_not, numpy.nansum, numpy.ones across 2 repos
# anchor_symbols: ['numpy.isnan', 'numpy.logical_not', 'numpy.nansum', 'numpy.ones', 'numpy.sum', 'numpy.where']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sberbank-ai-lab__LightAutoML::lightautoml/automl/blend.py::WeightedBlender._get_weighted_pred ---
def _get_weighted_pred(self, splitted_preds: Sequence[NumpyDataset], wts: Optional[np.ndarray]) -> NumpyDataset:
        length = len(splitted_preds)
        if wts is None:
            wts = np.ones(length, dtype=np.float32) / length

        weighted_pred = np.nansum([x.data * w for (x, w) in zip(splitted_preds, wts)], axis=0).astype(np.float32)

        not_nulls = np.sum(
            [np.logical_not(np.isnan(x.data).any(axis=1)) * w for (x, w) in zip(splitted_preds, wts)],
            axis=0,
        ).astype(np.float32)

        not_nulls = not_nulls[:, np.newaxis]

        weighted_pred /= not_nulls
        weighted_pred = np.where(not_nulls == 0, np.nan, weighted_pred)

        outp = splitted_preds[0].empty()
        outp.set_data(
            weighted_pred,
            ["WeightedBlend_{0}".format(x) for x in range(weighted_pred.shape[1])],
            NumericRole(np.float32, prob=self._outp_prob),
        )

        return outp

# --- from sb-ai-lab__LightAutoML::lightautoml/automl/blend.py::WeightedBlender._get_weighted_pred ---
def _get_weighted_pred(self, splitted_preds: Sequence[NumpyDataset], wts: Optional[np.ndarray]) -> NumpyDataset:
        length = len(splitted_preds)
        if wts is None:
            wts = np.ones(length, dtype=np.float32) / length

        weighted_pred = np.nansum([x.data * w for (x, w) in zip(splitted_preds, wts)], axis=0).astype(np.float32)

        not_nulls = np.sum(
            [np.logical_not(np.isnan(x.data).any(axis=1)) * w for (x, w) in zip(splitted_preds, wts)],
            axis=0,
        ).astype(np.float32)

        not_nulls = not_nulls[:, np.newaxis]

        weighted_pred /= not_nulls
        weighted_pred = np.where(not_nulls == 0, np.nan, weighted_pred)

        outp = splitted_preds[0].empty()
        outp.set_data(
            weighted_pred,
            self._class_mapping if self._class_mapping else list(range(weighted_pred.shape[1])),
            NumericRole(np.float32, prob=self._outp_prob),
        )

        return outp

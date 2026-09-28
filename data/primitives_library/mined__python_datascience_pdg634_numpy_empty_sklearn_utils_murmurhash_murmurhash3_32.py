# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg634::numpy.empty+sklearn.utils.murmurhash.murmurhash3_32
# name: numpy_sklearn_primitive
# summary: Uses numpy.empty, sklearn.utils.murmurhash.murmurhash3_32 across 2 repos
# anchor_symbols: ['numpy.empty', 'sklearn.utils.murmurhash.murmurhash3_32']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/transformers/categorical.py::CatIntersectstions._make_category ---
def _make_category(df: DataFrame, cols: Sequence[str]) -> np.ndarray:
        """Make hash for category interactions.

        Args:
            df: Input DataFrame
            cols: List of columns

        Returns:
            Hash np.ndarray.

        """
        res = np.empty((df.shape[0],), dtype=np.int32)

        for n, inter in enumerate(zip(*(df[x] for x in cols))):
            h = murmurhash3_32("_".join(map(str, inter)), seed=42)
            res[n] = h

        return res

# --- from sberbank-ai-lab__LightAutoML::lightautoml/transformers/categorical.py::CatIntersectstions._make_category ---
def _make_category(df: DataFrame, cols: Sequence[str]) -> np.ndarray:
        """Make hash for category interactions.

        Args:
            df: Input DataFrame
            cols: List of columns

        Returns:
            Hash np.ndarray.

        """
        res = np.empty((df.shape[0],), dtype=np.int32)

        for n, inter in enumerate(zip(*(df[x] for x in cols))):
            h = murmurhash3_32("_".join(map(str, inter)), seed=42)
            res[n] = h

        return res

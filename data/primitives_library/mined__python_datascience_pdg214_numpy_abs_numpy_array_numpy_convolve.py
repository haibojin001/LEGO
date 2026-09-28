# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg214::numpy.abs+numpy.array+numpy.convolve
# name: numpy_primitive
# summary: Uses numpy.abs, numpy.array, numpy.convolve, numpy.max across 3 repos
# anchor_symbols: ['numpy.abs', 'numpy.array', 'numpy.convolve', 'numpy.max', 'numpy.mean', 'numpy.median', 'numpy.min', 'numpy.ones', 'numpy.union1d']
# observed in 3 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'PIA-Group__BioSPPy', 'yzhao062__pyod']...

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/model/summary_algorithms.py::named_aggregate_summary ---
def named_aggregate_summary(series: pd.Series, key: str) -> dict:
    summary = {
        f"max_{key}": np.max(series),
        f"mean_{key}": np.mean(series),
        f"median_{key}": np.median(series),
        f"min_{key}": np.min(series),
    }

    return summary

# --- from PIA-Group__BioSPPy::biosppy/signals/emg.py::find_onsets ---
def find_onsets(signal=None, sampling_rate=1000., size=0.05, threshold=None):
    """Determine onsets of EMG pulses.

    Skips corrupted signal parts.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : float, optional
        Detection window size (seconds).
    threshold : float, optional
        Detection threshold.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # full-wave rectification
    fwlo = np.abs(signal)

    # smooth
    size = int(sampling_rate * size)
    mvgav, _ = st.smoother(signal=fwlo,
                           kernel='boxzen',
                           size=size,
                           mirror=True)

    # threshold
    if threshold is None:
        aux = np.abs(mvgav)
        threshold = 1.2 * np.mean(aux) + 2.0 * np.std(aux, ddof=1)

    # find onsets
    length = len(signal)
    start = np.nonzero(mvgav > threshold)[0]
    stop = np.nonzero(mvgav <= threshold)[0]

    onsets = np.union1d(np.intersect1d(start - 1, stop),
                        np.intersect1d(start + 1, stop))

    if np.any(onsets):
        if onsets[-1] >= length:
            onsets[-1] = length - 1

    return utils.ReturnTuple((onsets,), ('onsets',))

# --- from PIA-Group__BioSPPy::biosppy/signals/tools.py::signal_stats ---
def signal_stats(signal=None):
    """Compute various metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    mean : float
        Mean of the signal.
    median : float
        Median of the signal.
    min : float
        Minimum signal value.
    max : float
        Maximum signal value.
    max_amp : float
        Maximum absolute signal amplitude, in relation to the mean.
    var : float
        Signal variance (unbiased).
    std_dev : float
        Standard signal deviation (unbiased).
    abs_dev : float
        Mean absolute signal deviation around the median.
    kurtosis : float
        Signal kurtosis (unbiased).
    skew : float
        Signal skewness (unbiased).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # mean
    mean = np.mean(signal)

    # median
    median = np.median(signal)

    # min
    minVal = np.min(signal)

    # max
    maxVal = np.max(signal)

    # maximum amplitude
    maxAmp = np.abs(signal - mean).max()

    # variance
    sigma2 = signal.var(ddof=1)

    # standard deviation
    sigma = signal.std(ddof=1)

    # absolute deviation
    ad = np.mean(np.abs(signal - median))

    # kurtosis
    kurt = stats.kurtosis(signal, bias=False)

    # skweness
    skew = stats.skew(signal, bias=False)

    # output
    args = (mean, median, minVal, maxVal, maxAmp, sigma2, sigma, ad, kurt, skew)
    names = (
        "mean",
        "median",
        "min",
        "max",
        "max_amp",
        "var",
        "std_dev",
        "abs_dev",
        "kurtosis",
        "skewness",
    )

    return utils.ReturnTuple(args, names)

# --- from yzhao062__pyod::pyod/utils/ad_engine.py::ADEngine.analyze_results ---
def analyze_results(self, result: dict, X: Any = None,
                        top_k: int = 10) -> dict:
        """Analyze detection results.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        X : array-like or None
            Original training data for feature-level analysis.
        top_k : int
            Number of top anomalies to return.

        Returns
        -------
        analysis : dict
        """
        top_k = max(0, int(top_k))
        scores = result['scores_train']
        labels = result['labels_train']
        n_anomalies = int(labels.sum())

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_anomalies = [{'index': int(i), 'score': float(scores[i])}
                         for i in top_indices]

        score_dist = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
        }

        detector_name = result['plan'].get('detector_name', 'unknown')
        ratio = n_anomalies / len(labels) if len(labels) > 0 else 0
        summary = (
            "%d anomalies detected out of %d samples (%.1f%%) "
            "using %s. Scores range from %.4f to %.4f "
            "(mean=%.4f, std=%.4f). Threshold: %.4f."
            % (n_anomalies, len(labels), ratio * 100,
               detector_name,
               score_dist['min'], score_dist['max'],
               score_dist['mean'], score_dist['std'],
               result['threshold']))

        analysis = {
            'n_anomalies': n_anomalies,
            'anomaly_ratio': ratio,
            'score_distribution': score_dist,
            'top_anomalies': top_anomalies,
            'summary': summary,
        }

        if X is not None:
            fi = compute_feature_importance(result, X)
            if fi is not None:
                analysis['feature_importance'] = fi

        return analysis

# --- from yzhao062__pyod::pyod/utils/ad_engine.py::ADEngine.analyze ---
def analyze(self, state: InvestigationState) -> InvestigationState:
        """Analyze detection results with quality assessment.

        Computes per-detector analysis, consensus analysis, quality
        metrics (separation, agreement, stability), and selects
        the best detector.

        Parameters
        ----------
        state : InvestigationState

        Returns
        -------
        state : InvestigationState
        """
        self._require_phase(state, 'detected')
        from .investigation import _make_history_entry

        state.phase = 'analyzed'

        # All-error path
        successful = [r for r in state.results
                      if r['status'] == 'success']
        if not successful:
            state.analysis = None
            state.quality = {
                'separation': 0.0, 'agreement': 0.0,
                'stability': 0.0, 'overall': 0.0,
                'verdict': 'low',
                'explanation': 'All detectors failed.',
            }
            state.next_action = {
                'action': 'confirm_with_user',
                'reason': 'All detectors failed. Check data format '
                          'or try a different detector family.',
            }
            state.history.append(_make_history_entry(
                'analyzed', 'analyze', state.iteration,
                'All detectors failed'))
            return state

        # Per-detector analysis (aligned with state.results)
        per_det = []
        for r in state.results:
            if r['status'] == 'success':
                try:
                    a = self.analyze_results(r, X=state.data)
                except Exception as exc:
                    logger.warning(
                        'analyze_results failed for %s with %s: %s',
                        r.get('detector_name', '<unknown>'),
                        type(exc).__name__, exc)
                    a = None
                per_det.append(a)
            else:
                per_det.append(None)

        # Consensus analysis (lightweight, not via analyze_results)
        c = state.consensus
        c_scores = c['scores']
        c_labels = c['labels']
        n_anomalies = int(c_labels.sum())
        n_samples = len(c_labels)
        top_k = min(10, n_samples)
        top_indices = np.argsort(c_scores)[::-1][:top_k]
        consensus_analysis = {
            'n_anomalies': n_anomalies,
            'anomaly_ratio': n_anomalies / max(n_samples, 1),
            'score_distribution': {
                'mean': float(np.mean(c_scores)),
                'std': float(np.std(c_scores)),
                'min': float(np.min(c_scores)),
                'max': float(np.max(c_scores)),
                'median': float(np.median(c_scores)),
                'q25': float(np.percentile(c_scores, 25)),
                'q75': float(np.percentile(c_scores, 75)),
            },
            'top_anomalies': [
                {'index': int(i), 'score': float(c_scores[i])}
                for i in top_indices],
            'summary': '%d anomalies detected out of %d samples '
                       '(%.1f%%) by consensus of %d detectors.'
                       % (n_anomalies, n_samples,
                          100 * n_anomalies / max(n_samples, 1),
                          c['n_detectors']),
        }

        # Best detector selection
        best_idx = select_best_detector(
            state.results, c_scores)

        state.analysis = {
            'consensus_analysis': consensus_analysis,
            'per_detector_analysis': per_det,
            'best_detector': state.results[best_idx]['detector_name'],
            'best_detector_index': best_idx,
            'summary': consensus_analysis['summary'],
        }

        # Quality metrics
        state.quality = compute_quality(
            c_scores, c_labels, state.results, c)
        state.analysis['summary'] += (
            ' Quality: %s (%.2f).'
            % (state.quality['verdict'], state.quality['overall']))

        # Next action based on quality
        if state.quality['overall'] >= 0.4:
            state.next_action = {
                'action': 'report_to_user',
                'reason': 'Results ready (quality=%s, %.2f).'
                          % (state.quality['verdict'],
                             state.quality['overall']),
                'summary': state.analysis['summary'],
                'confidence': state.quality['overall'],
            }
        else:
            state.next_action = {
                'action': 'iterate',
                'reason': 'Low result quality (%.2f). Consider '
                          'trying different detectors.'
                          % state.quality['overall'],
                'suggestion': 'Exclude lowest-agreement detector '
                              'and re-run.',
            }

        state.history.append(_make_history_entry(
            'analyzed', 'analyze', state.iteration,
            'Quality: %s (%.2f)' % (
                state.quality['verdict'],
                state.quality['overall'])))
        return state

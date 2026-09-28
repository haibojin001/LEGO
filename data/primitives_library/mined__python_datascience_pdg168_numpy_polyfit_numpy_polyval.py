# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg168::numpy.polyfit+numpy.polyval
# name: numpy_primitive
# summary: Uses numpy.polyfit, numpy.polyval across 3 repos
# anchor_symbols: ['numpy.polyfit', 'numpy.polyval']
# observed in 3 repos: ['fraunhoferportugal__tsfel', 'mwaskom__seaborn', 'shashankvemuri__Finance']...

# --- from shashankvemuri__Finance::ta_functions.py::LINEARREG.linreg ---
def linreg(x):
        return np.polyval(np.polyfit(idx, x, 1), idx)[-1]

# --- from mwaskom__seaborn::seaborn/regression.py::_RegressionPlotter.fit_poly.reg_func ---
def reg_func(_x, _y):
            return np.polyval(np.polyfit(_x, _y, order), grid)

# --- from mwaskom__seaborn::tests/_core/test_groupby.py::test_apply_mutate_columns.polyfit ---
def polyfit(df):
        fit = np.polyfit(df["x"], df["y"], 1)
        hat = np.polyval(fit, xx)
        hats.append(hat)
        return pd.DataFrame(dict(x=xx, y=hat))

# --- from shashankvemuri__Finance::ta_functions.py::LINEARREG ---
def LINEARREG(close, timeperiod=14):
    """
    Linear Regression.
    A statistical way to predict future prices based on past prices.
    """
    idx = np.arange(timeperiod)
    def linreg(x):
        return np.polyval(np.polyfit(idx, x, 1), idx)[-1]
    return close.rolling(window=timeperiod).apply(linreg, raw=True)

# --- from fraunhoferportugal__tsfel::tests/complexity_measures_intuition.py::plot_dfa ---
def plot_dfa(info, scale, fluctuations, ax):
    """Plot log-log visualization of the detrended fluctuation analysis
    (DFA)."""

    polyfit = np.polyfit(np.log2(scale), np.log2(fluctuations), 1)
    fluctfit = 2 ** np.polyval(polyfit, np.log2(scale))
    ax.loglog(scale, fluctuations, "o", c="#90A4AE")
    ax.loglog(
        scale,
        fluctfit,
        c="#E91E63",
        label=r"$\alpha$ = {:.3f}".format(info["Alpha"]),
    )

    return ax

# --- from fraunhoferportugal__tsfel::tsfel/feature_extraction/features_utils.py::calc_rms ---
def calc_rms(signal, window):
    """Windowed Root Mean Square (RMS) with linear detrending.

    Parameters
    ----------
    signal: nd-array
        Signal
    window: int
        Length of the window in which RMS will be calculated

    Returns
    -------
    rms : nd-array
        RMS data in each window with length len(signal)//window
    """
    num_windows = len(signal) // window
    rms = np.zeros(num_windows)

    for idx in np.arange(num_windows):
        start_idx = idx * window
        end_idx = start_idx + window
        windowed_signal = signal[start_idx:end_idx]

        coeff = np.polyfit(np.arange(window), windowed_signal, 1)
        detrended_window = windowed_signal - np.polyval(coeff, np.arange(window))
        rms[idx] = np.sqrt(np.mean(detrended_window**2))

    return rms

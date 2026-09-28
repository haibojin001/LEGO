# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg496::numpy.ones_like+numpy.testing.assert_array_equal
# name: numpy_primitive
# summary: Uses numpy.ones_like, numpy.testing.assert_array_equal across 2 repos
# anchor_symbols: ['numpy.ones_like', 'numpy.testing.assert_array_equal']
# observed in 2 repos: ['CamDavidsonPilon__lifelines', 'wilsonrljr__sysidentpy']...

# --- from CamDavidsonPilon__lifelines::lifelines/tests/utils/test_utils.py::test_survival_events_from_table_no_ties ---
def test_survival_events_from_table_no_ties():
    T, C = np.array([1, 2, 3, 4, 4, 5]), np.array([1, 0, 1, 1, 0, 1])
    d = utils.survival_table_from_events(T, C)
    T_, C_, W_ = utils.survival_events_from_table(d[["censored", "observed"]])
    npt.assert_array_equal(T, T_)
    npt.assert_array_equal(C, C_)
    npt.assert_array_equal(W_, np.ones_like(T))

# --- from wilsonrljr__sysidentpy::sysidentpy/multiobjective_parameter_estimation/tests/test_mo_estimators.py::test_build_system_data_branching ---
def test_build_system_data_branching():
    y = np.arange(4).reshape(-1, 1)
    gain_sample = np.ones_like(y)
    static_sample = 2 * np.ones_like(y)

    model_gain_only = AILS(static_gain=True, static_function=False)
    parts_gain = model_gain_only.build_system_data(y, gain_sample, static_sample)
    assert len(parts_gain) == 2
    assert_array_equal(parts_gain[0], y)
    assert_array_equal(parts_gain[1], gain_sample)

    model_function_only = AILS(static_gain=False, static_function=True)
    parts_function = model_function_only.build_system_data(
        y, gain_sample, static_sample
    )
    assert len(parts_function) == 2
    assert_array_equal(parts_function[0], y)
    assert_array_equal(parts_function[1], static_sample)

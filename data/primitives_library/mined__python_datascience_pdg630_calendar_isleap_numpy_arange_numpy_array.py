# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg630::calendar.isleap+numpy.arange+numpy.array
# name: calendar_numpy_primitive
# summary: Uses calendar.isleap, numpy.arange, numpy.array, pandas.Timedelta across 2 repos
# anchor_symbols: ['calendar.isleap', 'numpy.arange', 'numpy.array', 'pandas.Timedelta']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/automl/presets/utils.py::change_datetime ---
def change_datetime(feature_datetime, key, value):
    assert key in ["year", "month", "dayofweek"]
    MAX_DAY = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    changed = []
    if key == "year":
        year = value
        for i in feature_datetime:
            if i.month == 2 and i.day == 29 and not calendar.isleap(year):
                i -= pd.Timedelta(1, "d")
            changed.append(i.replace(year=year))
    if key == "month":
        month = value
        assert month in np.arange(1, 13)
        for i in feature_datetime:
            if i.day > MAX_DAY[month]:
                i -= pd.Timedelta(i.day - MAX_DAY[month], "d")
                if month == 2 and i.day == 28 and calendar.isleap(i.year):
                    i += pd.Timedelta(1, "d")
            changed.append(i.replace(month=month))
    if key == "dayofweek":
        dayofweek = value
        assert value in np.arange(7)
        for i in feature_datetime:
            i += pd.Timedelta(dayofweek - i.dayofweek, "d")
            changed.append(i)
    return np.array(changed)

# --- from sberbank-ai-lab__LightAutoML::lightautoml/automl/presets/utils.py::change_datetime ---
def change_datetime(feature_datetime, key, value):
    assert key in ["year", "month", "dayofweek"]
    MAX_DAY = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    changed = []
    if key == "year":
        year = value
        for i in feature_datetime:
            if i.month == 2 and i.day == 29 and not calendar.isleap(year):
                i -= pd.Timedelta(1, "d")
            changed.append(i.replace(year=year))
    if key == "month":
        month = value
        assert month in np.arange(1, 13)
        for i in feature_datetime:
            if i.day > MAX_DAY[month]:
                i -= pd.Timedelta(i.day - MAX_DAY[month], "d")
                if month == 2 and i.day == 28 and calendar.isleap(i.year):
                    i += pd.Timedelta(1, "d")
            changed.append(i.replace(month=month))
    if key == "dayofweek":
        dayofweek = value
        assert value in np.arange(7)
        for i in feature_datetime:
            i += pd.Timedelta(dayofweek - i.dayofweek, "d")
            changed.append(i)
    return np.array(changed)

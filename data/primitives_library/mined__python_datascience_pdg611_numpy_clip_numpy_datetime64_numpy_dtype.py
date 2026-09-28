# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg611::numpy.clip+numpy.datetime64+numpy.dtype
# name: numpy_pandas_primitive
# summary: Uses numpy.clip, numpy.datetime64, numpy.dtype, pandas.read_csv across 2 repos
# anchor_symbols: ['numpy.clip', 'numpy.datetime64', 'numpy.dtype', 'pandas.read_csv', 'sklearn.model_selection.train_test_split']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::tests/conftest.py::sampled_app_train_test ---
def sampled_app_train_test(nrows=None):
    data = pd.read_csv(
        "./examples/data/sampled_app_train.csv",
        nrows=nrows,
    )

    data["BIRTH_DATE"] = np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    data["EMP_DATE"] = np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(
        np.dtype("timedelta64[D]")
    )
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED", "SK_ID_CURR"], axis=1, inplace=True)

    data["__fold__"] = np.random.randint(0, 5, len(data))

    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=RANDOM_STATE)

    return train_data, test_data

# --- from sberbank-ai-lab__LightAutoML::tests/conftest.py::sampled_app_train_test ---
def sampled_app_train_test(nrows=None):
    data = pd.read_csv(
        "./examples/data/sampled_app_train.csv",
        usecols=[
            "TARGET",
            "NAME_CONTRACT_TYPE",
            "AMT_CREDIT",
            "NAME_TYPE_SUITE",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
        ],
        nrows=nrows,
    )

    data["BIRTH_DATE"] = np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    data["EMP_DATE"] = np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(
        np.dtype("timedelta64[D]")
    )
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

    data["__fold__"] = np.random.randint(0, 5, len(data))

    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=RANDOM_STATE)

    return train_data, test_data

# --- from sb-ai-lab__LightAutoML::tests/conftest.py::uplift_data_train_test ---
def uplift_data_train_test(sampled_app_roles, nrows=None):
    data = pd.read_csv(
        "./examples/data/sampled_app_train.csv",
        nrows=nrows,
    )
    sampled_app_roles["treatment"] = "CODE_GENDER"

    data["BIRTH_DATE"] = (np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))).astype(
        str
    )
    data["EMP_DATE"] = (
        np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
    ).astype(str)
    data["report_dt"] = np.datetime64("2018-01-01")
    data["constant"] = 1
    data["allnan"] = np.nan
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)
    data["CODE_GENDER"] = (data["CODE_GENDER"] == "M").astype(int)
    data["__fold__"] = np.random.randint(0, 5, len(data))

    stratify_value = data[get_target_name(sampled_app_roles)] + 10 * data[sampled_app_roles["treatment"]]
    train, test = train_test_split(data, test_size=3000, stratify=stratify_value, random_state=42)
    test_target, test_treatment = (
        test[get_target_name(sampled_app_roles)].values.ravel(),
        test[sampled_app_roles["treatment"]].values.ravel(),
    )

    return train, test, test_target, test_treatment

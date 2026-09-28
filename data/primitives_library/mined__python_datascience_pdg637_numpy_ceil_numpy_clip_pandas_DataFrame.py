# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg637::numpy.ceil+numpy.clip+pandas.DataFrame
# name: numpy_pandas_primitive
# summary: Uses numpy.ceil, numpy.clip, pandas.DataFrame across 2 repos
# anchor_symbols: ['numpy.ceil', 'numpy.clip', 'pandas.DataFrame']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/report/report_deco.py::f1_score_w_co ---
def f1_score_w_co(input_data, min_co=0.01, max_co=0.99, step=0.01):
    data = input_data.copy()
    data["y_pred"] = np.clip(np.ceil(data["y_pred"].values / step) * step, min_co, max_co)

    pos = data["y_true"].sum()
    neg = data["y_true"].shape[0] - pos

    grp = pd.DataFrame(data).groupby("y_pred")["y_true"].agg(["sum", "count"])
    grp.sort_index(inplace=True)

    grp["fp"] = grp["sum"].cumsum()
    grp["tp"] = pos - grp["fp"]
    grp["tn"] = (grp["count"] - grp["sum"]).cumsum()
    grp["fn"] = neg - grp["tn"]

    grp["pr"] = grp["tp"] / (grp["tp"] + grp["fp"])
    grp["rec"] = grp["tp"] / (grp["tp"] + grp["fn"])

    grp["f1_score"] = 2 * (grp["pr"] * grp["rec"]) / (grp["pr"] + grp["rec"])

    best_score = grp["f1_score"].max()
    best_co = grp.index.values[grp["f1_score"] == best_score].mean()

    return best_score, best_co

# --- from sberbank-ai-lab__LightAutoML::lightautoml/report/report_deco.py::f1_score_w_co ---
def f1_score_w_co(input_data, min_co=0.01, max_co=0.99, step=0.01):
    data = input_data.copy()
    data["y_pred"] = np.clip(np.ceil(data["y_pred"].values / step) * step, min_co, max_co)

    pos = data["y_true"].sum()
    neg = data["y_true"].shape[0] - pos

    grp = pd.DataFrame(data).groupby("y_pred")["y_true"].agg(["sum", "count"])
    grp.sort_index(inplace=True)

    grp["fp"] = grp["sum"].cumsum()
    grp["tp"] = pos - grp["fp"]
    grp["tn"] = (grp["count"] - grp["sum"]).cumsum()
    grp["fn"] = neg - grp["tn"]

    grp["pr"] = grp["tp"] / (grp["tp"] + grp["fp"])
    grp["rec"] = grp["tp"] / (grp["tp"] + grp["fn"])

    grp["f1_score"] = 2 * (grp["pr"] * grp["rec"]) / (grp["pr"] + grp["rec"])

    best_score = grp["f1_score"].max()
    best_co = grp.index.values[grp["f1_score"] == best_score].mean()

    # print((y_pred < best_co).mean())

    return best_score, best_co

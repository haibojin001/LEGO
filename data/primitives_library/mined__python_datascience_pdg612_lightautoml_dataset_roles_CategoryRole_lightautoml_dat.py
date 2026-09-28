# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg612::lightautoml.dataset.roles.CategoryRole+lightautoml.dataset.roles.DatetimeRole+lightautoml.dataset.roles.FoldsRole
# name: lightautoml_primitive
# summary: Uses lightautoml.dataset.roles.CategoryRole, lightautoml.dataset.roles.DatetimeRole, lightautoml.dataset.roles.FoldsRole, lightautoml.dataset.roles.NumericRole across 2 repos
# anchor_symbols: ['lightautoml.dataset.roles.CategoryRole', 'lightautoml.dataset.roles.DatetimeRole', 'lightautoml.dataset.roles.FoldsRole', 'lightautoml.dataset.roles.NumericRole', 'lightautoml.dataset.roles.TargetRole']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::tests/conftest.py::sampled_app_roles ---
def sampled_app_roles():
    return {
        TargetRole(): "TARGET",
        CategoryRole(dtype=str): ["NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE"],
        NumericRole(np.float32): ["AMT_CREDIT", "AMT_GOODS_PRICE"],
        DatetimeRole(seasonality=["y", "m", "wd"]): ["BIRTH_DATE", "EMP_DATE"],
        FoldsRole(): "__fold__",
    }

# --- from sberbank-ai-lab__LightAutoML::tests/conftest.py::sampled_app_roles ---
def sampled_app_roles():
    return {
        TargetRole(): "TARGET",
        CategoryRole(dtype=str): ["NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE"],
        NumericRole(np.float32): ["AMT_CREDIT", "AMT_GOODS_PRICE"],
        DatetimeRole(seasonality=["y", "m", "wd"]): ["BIRTH_DATE", "EMP_DATE"],
        FoldsRole(): "__fold__",
    }

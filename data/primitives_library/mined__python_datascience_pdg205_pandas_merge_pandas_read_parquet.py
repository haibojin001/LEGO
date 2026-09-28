# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg205::pandas.merge+pandas.read_parquet
# name: pandas_primitive
# summary: Uses pandas.merge, pandas.read_parquet across 2 repos
# anchor_symbols: ['pandas.merge', 'pandas.read_parquet']
# observed in 2 repos: ['microsoft__RD-Agent', 'predict-idlab__tsflex']...

# --- from predict-idlab__tsflex::tests/utils.py::dummy_data ---
def dummy_data() -> pd.DataFrame:
    df1 = pd.read_parquet(proj_dir + "/examples/data/empatica/gsr.parquet")
    df2 = pd.read_parquet(proj_dir + "/examples/data/empatica/tmp.parquet")
    df3 = pd.read_parquet(proj_dir + "/examples/data/empatica/acc.parquet")
    df = pd.merge(df1, df2, how="inner", on="timestamp")
    df = pd.merge(df, df3, how="inner", on="timestamp")
    df.set_index("timestamp", inplace=True)
    return df

# --- from microsoft__RD-Agent::rdagent/scenarios/kaggle/experiment/templates/optiver-realized-volatility-prediction/fea_share_preprocess.py::prepreprocess ---
def prepreprocess():
    # Load the training data
    train_df = pd.read_csv("/kaggle/input/train.csv")

    # Load book and trade data
    book_train = pd.read_parquet("/kaggle/input/book_train.parquet")
    trade_train = pd.read_parquet("/kaggle/input/trade_train.parquet")

    # Merge book and trade data with train_df
    merged_df = pd.merge(train_df, book_train, on=["stock_id", "time_id"], how="left")
    merged_df = pd.merge(merged_df, trade_train, on=["stock_id", "time_id"], how="left")

    # Split the data
    X = merged_df.drop(["target"], axis=1)
    y = merged_df["target"]

    print(X.columns.to_list())

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.columns.to_list())

    return X_train, X_valid, y_train, y_valid

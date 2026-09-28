# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg92::datetime.date+pandas.to_datetime
# name: datetime_pandas_primitive
# summary: Uses datetime.date, pandas.to_datetime across 2 repos
# anchor_symbols: ['datetime.date', 'pandas.to_datetime']
# observed in 2 repos: ['akfamily__akshare', 'feature-engine__feature_engine']...

# --- from feature-engine__feature_engine::tests/test_datetime/test_datetime_ordinal.py::test_datetime_ordinal_with_start_date_datetime_object ---
def test_datetime_ordinal_with_start_date_datetime_object(df_datetime_ordinal):
    start_date_obj = datetime.date(2023, 1, 1)
    transformer = DatetimeOrdinal(variables=["date_col_1"], start_date=start_date_obj)
    X_transformed = transformer.fit_transform(df_datetime_ordinal)

    start_ordinal = pd.to_datetime(start_date_obj).toordinal()
    expected_ordinal = pd.Series(
        [d.toordinal() - start_ordinal + 1 for d in df_datetime_ordinal["date_col_1"]],
        name="date_col_1_ordinal",
    )

    pd.testing.assert_series_equal(
        X_transformed["date_col_1_ordinal"], expected_ordinal
    )

# --- from akfamily__akshare::akshare/tool/trade_date_hist.py::tool_trade_date_hist_sina ---
def tool_trade_date_hist_sina() -> pd.DataFrame:
    """
    新浪财经-交易日历-历史数据
    https://finance.sina.com.cn/realstock/company/klc_td_sh.txt
    :return: 交易日历
    :rtype: pandas.DataFrame
    """
    url = "https://finance.sina.com.cn/realstock/company/klc_td_sh.txt"
    r = requests.get(url)
    js_code = py_mini_racer.MiniRacer()
    js_code.eval(hk_js_decode)
    dict_list = js_code.call("d", r.text.split("=")[1].split(";")[0].replace('"', ""))
    temp_df = pd.DataFrame(dict_list)
    temp_df.columns = ["trade_date"]
    temp_df["trade_date"] = pd.to_datetime(temp_df["trade_date"]).dt.date
    temp_list = temp_df["trade_date"].to_list()
    # 该日期是交易日，但是在新浪返回的交易日历缺失该日期，这里补充上
    temp_list.append(datetime.date(year=1992, month=5, day=4))
    temp_list.sort()
    temp_df = pd.DataFrame(temp_list, columns=["trade_date"])
    return temp_df

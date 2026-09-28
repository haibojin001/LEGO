# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg91::pandas.to_datetime
# name: pandas_primitive
# summary: Uses pandas.to_datetime across 3 repos
# anchor_symbols: ['pandas.to_datetime']
# observed in 3 repos: ['akfamily__akshare', 'deepchecks__deepchecks', 'lux-org__lux']...

# --- from lux-org__lux::tests/test_pandas_coverage.py::test_qcut ---
def test_qcut(global_var):
    df = pd.read_csv("lux/data/car.csv")
    df["Year"] = pd.to_datetime(df["Year"], format="%Y")
    df["Weight"] = pd.qcut(df["Weight"], q=3)
    df._ipython_display_()

# --- from lux-org__lux::tests/test_pandas_coverage.py::test_read_multi_dtype ---
def test_read_multi_dtype(global_var):
    url = "https://github.com/lux-org/lux-datasets/blob/master/data/car-data.xls?raw=true"
    df = pd.read_excel(url)
    with pytest.warns(UserWarning, match="mixed type") as w:
        df._ipython_display_()
        assert "df['Car Type'] = df['Car Type'].astype(str)" in str(w[-1].message)

# --- from deepchecks__deepchecks::deepchecks/tabular/datasets/classification/phishing.py::UrlDatasetProcessor._shared_preprocess ---
def _shared_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[_DATE_COL] = pd.to_datetime(
            df[_DATE_COL], format='%Y-%m-%d')
        df = df.set_index(keys=_DATE_COL, drop=True)
        df = df.drop(_NON_FEATURES, axis=1)
        df = pd.get_dummies(df, columns=['ext'])
        return df

# --- from akfamily__akshare::akshare/stock/stock_us_sina.py::stock_us_daily ---
def stock_us_daily(symbol: str = "FB", adjust: str = "") -> pd.DataFrame:
    """
    新浪财经-美股
    https://finance.sina.com.cn/stock/usstock/sector.shtml
    备注：
    1. CIEN 新浪复权因子错误
    2. AI 新浪复权因子错误, 该股票刚上市未发生复权, 但是返回复权因子
    :param symbol: 可以使用 get_us_stock_name 获取
    :type symbol: str
    :param adjust: "": 返回未复权的数据 ; qfq: 返回前复权后的数据; qfq-factor: 返回前复权因子和调整;
    :type adjust: str
    :return: 指定 adjust 的数据
    :rtype: pandas.DataFrame
    """
    url = f"https://finance.sina.com.cn/staticdata/us/{symbol}"
    res = requests.get(url)
    js_code = py_mini_racer.MiniRacer()
    js_code.eval(zh_js_decode)
    dict_list = js_code.call("d", res.text.split("=")[1].split(";")[0].replace('"', ""))
    data_df = pd.DataFrame(dict_list)
    data_df["date"] = pd.to_datetime(data_df["date"]).dt.date
    data_df.index = pd.to_datetime(data_df["date"])
    del data_df["amount"]
    del data_df["date"]
    data_df = data_df.astype("float")
    url = us_sina_stock_hist_qfq_url.format(symbol)
    res = requests.get(url)
    qfq_factor_df = pd.DataFrame(eval(res.text.split("=")[1].split("\n")[0])["data"])
    qfq_factor_df.rename(
        columns={
            "c": "adjust",
            "d": "date",
            "f": "qfq_factor",
        },
        inplace=True,
    )
    qfq_factor_df.index = pd.to_datetime(qfq_factor_df["date"])
    del qfq_factor_df["date"]

    # 处理复权因子
    temp_date_range = pd.date_range("1900-01-01", qfq_factor_df.index[0].isoformat())
    temp_df = pd.DataFrame(range(len(temp_date_range)), temp_date_range)
    new_range = pd.merge(
        temp_df, qfq_factor_df, left_index=True, right_index=True, how="left"
    )
    new_range = new_range.ffill()
    new_range = new_range.iloc[:, [1, 2]]

    if adjust == "qfq":
        if len(new_range) == 1:
            new_range.index = [pd.to_datetime(str(data_df.index.date[0]))]
        temp_df = pd.merge(
            data_df, new_range, left_index=True, right_index=True, how="left"
        )
        try:
            # try for pandas >= 2.1.0
            temp_df.ffill(inplace=True)
        except Exception:
            try:
                # try for pandas < 2.1.0
                temp_df.fillna(method="ffill", inplace=True)
            except Exception as e:
                print("Error:", e)
        try:
            # try for pandas >= 2.1.0
            temp_df.bfill(inplace=True)
        except Exception:
            try:
                # try for pandas < 2.1.0
                temp_df.fillna(method="bfill", inplace=True)
            except Exception as e:
                print("Error:", e)

        temp_df = temp_df.astype(float)
        temp_df["open"] = temp_df["open"] * temp_df["qfq_factor"] + temp_df["adjust"]
        temp_df["high"] = temp_df["high"] * temp_df["qfq_factor"] + temp_df["adjust"]
        temp_df["close"] = temp_df["close"] * temp_df["qfq_factor"] + temp_df["adjust"]
        temp_df["low"] = temp_df["low"] * temp_df["qfq_factor"] + temp_df["adjust"]
        temp_df = temp_df.apply(lambda x: round(x, 4))
        temp_df = temp_df.astype("float")
        # 处理复权因子错误的情况-开始
        check_df = temp_df[["open", "high", "low", "close"]].copy()
        check_df.dropna(inplace=True)
        if check_df.empty:
            data_df.reset_index(inplace=True)
            return data_df
        # 处理复权因子错误的情况-结束
        result_data = temp_df.iloc[:, :-2]
        result_data.reset_index(inplace=True)
        return result_data

    if adjust == "qfq-factor":
        qfq_factor_df.reset_index(inplace=True)
        return qfq_factor_df

    if adjust == "":
        data_df.reset_index(inplace=True)
        return data_df

# --- from akfamily__akshare::akshare/stock/stock_zh_b_sina.py::stock_zh_b_minute ---
def stock_zh_b_minute(
    symbol: str = "sh900901", period: str = "1", adjust: str = ""
) -> pd.DataFrame:
    """
    股票及股票指数历史行情数据-分钟数据
    https://finance.sina.com.cn/realstock/company/sh900901/nc.shtml
    :param symbol: sh900901
    :type symbol: str
    :param period: 1, 5, 15, 30, 60 分钟的数据
    :type period: str
    :param adjust: 默认为空: 返回不复权的数据; qfq: 返回前复权后的数据; hfq: 返回后复权后的数据;
    :type adjust: str
    :return: specific data
    :rtype: pandas.DataFrame
    """
    url = (
        "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData"
    )
    params = {
        "symbol": symbol,
        "scale": period,
        "datalen": "1970",
    }
    r = requests.get(url, params=params)
    temp_df = pd.DataFrame(json.loads(r.text.split("=(")[1].split(");")[0])).iloc[:, :6]
    if temp_df.empty:
        return pd.DataFrame()
    try:
        stock_zh_b_daily(symbol=symbol, adjust="qfq")
    except:  # noqa: E722
        return temp_df

    if adjust == "":
        return temp_df

    if adjust == "qfq":
        temp_df[["date", "time"]] = temp_df["day"].str.split(" ", expand=True)
        # 处理没有最后一分钟的情况
        need_df = temp_df[
            [
                True if "09:31:00" <= item <= "15:00:00" else False
                for item in temp_df["time"]
            ]
        ]
        need_df.drop_duplicates(subset=["date"], keep="last", inplace=True)
        need_df.index = pd.to_datetime(need_df["date"])
        stock_zh_b_daily_qfq_df = stock_zh_b_daily(symbol=symbol, adjust="qfq")
        stock_zh_b_daily_qfq_df.index = pd.to_datetime(stock_zh_b_daily_qfq_df["date"])
        result_df = stock_zh_b_daily_qfq_df.iloc[-len(need_df) :, :]["close"].astype(
            float
        ) / need_df["close"].astype(float)
        temp_df.index = pd.to_datetime(temp_df["date"])
        merged_df = pd.merge(temp_df, result_df, left_index=True, right_index=True)
        merged_df["open"] = merged_df["open"].astype(float) * merged_df["close_y"]
        merged_df["high"] = merged_df["high"].astype(float) * merged_df["close_y"]
        merged_df["low"] = merged_df["low"].astype(float) * merged_df["close_y"]
        merged_df["close"] = merged_df["close_x"].astype(float) * merged_df["close_y"]
        temp_df = merged_df[["day", "open", "high", "low", "close", "volume"]]
        temp_df.reset_index(drop=True, inplace=True)
        return temp_df
    if adjust == "hfq":
        temp_df[["date", "time"]] = temp_df["day"].str.split(" ", expand=True)
        # 处理没有最后一分钟的情况
        need_df = temp_df[
            [
                True if "09:31:00" <= item <= "15:00:00" else False
                for item in temp_df["time"]
            ]
        ]
        need_df.drop_duplicates(subset=["date"], keep="last", inplace=True)
        need_df.index = pd.to_datetime(need_df["date"])
        stock_zh_b_daily_hfq_df = stock_zh_b_daily(symbol=symbol, adjust="hfq")
        stock_zh_b_daily_hfq_df.index = pd.to_datetime(stock_zh_b_daily_hfq_df["date"])
        result_df = stock_zh_b_daily_hfq_df.iloc[-len(need_df) :, :]["close"].astype(
            float
        ) / need_df["close"].astype(float)
        temp_df.index = pd.to_datetime(temp_df["date"])
        merged_df = pd.merge(temp_df, result_df, left_index=True, right_index=True)
        merged_df["open"] = merged_df["open"].astype(float) * merged_df["close_y"]
        merged_df["high"] = merged_df["high"].astype(float) * merged_df["close_y"]
        merged_df["low"] = merged_df["low"].astype(float) * merged_df["close_y"]
        merged_df["close"] = merged_df["close_x"].astype(float) * merged_df["close_y"]
        temp_df = merged_df[["day", "open", "high", "low", "close", "volume"]]
        temp_df.reset_index(drop=True, inplace=True)
        return temp_df

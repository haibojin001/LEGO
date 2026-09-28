# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg84::pandas.DataFrame+pandas.to_datetime+pandas.to_numeric
# name: pandas_requests_primitive
# summary: Uses pandas.DataFrame, pandas.to_datetime, pandas.to_numeric, requests.get across 4 repos
# anchor_symbols: ['pandas.DataFrame', 'pandas.to_datetime', 'pandas.to_numeric', 'requests.get']
# observed in 4 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'akfamily__akshare', 'lux-org__lux', 'scikit-mobility__scikit-mobility']...

# --- from Data-Centric-AI-Community__fg-data-profiling::tests/unit/test_sensitive.py::df ---
def df():
    df = pd.DataFrame(
        {
            "name": ["John Doe", "Marco Polo", "Louis Brandeis", "William Douglas"],
            "year": [1965, 1271, 1916, 1975],
            "tf": [True, False, False, True],
            "date": pd.to_datetime(
                [datetime.now() - timedelta(days=i) for i in range(4)]
            ),
        }
    )
    return df

# --- from lux-org__lux::lux/executor/PandasExecutor.py::PandasExecutor._is_datetime_string ---
def _is_datetime_string(series):
        if series.dtype == object:
            not_numeric = False
            try:
                pd.to_numeric(series)
            except Exception as e:
                not_numeric = True

            datetime_col = None
            if not_numeric:
                try:
                    datetime_col = pd.to_datetime(series)
                except Exception as e:
                    return False
            if datetime_col is not None:
                return True
        return False

# --- from akfamily__akshare::akshare/fund/fund_aum_em.py::fund_aum_trend_em ---
def fund_aum_trend_em() -> pd.DataFrame:
    """
    东方财富-基金-基金市场管理规模走势图
    https://fund.eastmoney.com/Company/default.html
    :return: 基金市场管理规模走势图
    :rtype: pandas.DataFrame
    """
    url = "https://fund.eastmoney.com/Company/home/GetFundTotalScaleForChart"
    payload = {"fundType": "0"}
    r = requests.get(url, data=payload)
    data_json = r.json()
    temp_df = pd.DataFrame()
    temp_df["date"] = data_json["x"]
    temp_df["value"] = data_json["y"]
    temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce").dt.date
    temp_df["value"] = pd.to_numeric(temp_df["value"], errors="coerce")
    return temp_df

# --- from akfamily__akshare::akshare/index/index_option_qvix.py::index_option_50etf_qvix ---
def index_option_50etf_qvix() -> pd.DataFrame:
    """
    50ETF 期权波动率指数 QVIX
    http://1.optbbs.com/s/vix.shtml?50ETF
    :return: 50ETF 期权波动率指数 QVIX
    :rtype: pandas.DataFrame
    """
    temp_df = __get_optbbs_daily().iloc[:, :5]
    temp_df.columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
    ]
    temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce").dt.date
    temp_df["open"] = pd.to_numeric(temp_df["open"], errors="coerce")
    temp_df["high"] = pd.to_numeric(temp_df["high"], errors="coerce")
    temp_df["low"] = pd.to_numeric(temp_df["low"], errors="coerce")
    temp_df["close"] = pd.to_numeric(temp_df["close"], errors="coerce")
    return temp_df

# --- from scikit-mobility__scikit-mobility::skmob/data/datasets/taxi_san_francisco/taxi_san_francisco.py::taxi_san_francisco.prepare ---
def prepare(self, f_names):

        # keep only the filename of one taxi, e.g., new_abboip.txt
        fs = [f for f in f_names if 'new_abboip.txt' in f]

        # adjust the dataset in order to obtain a TrajDataFrame
        
        # to convert the datetime from UNIX to UTC
        mydateparser = lambda x: pandas.to_datetime(x, unit='s') + timedelta(minutes=-7*60)
        
        
        #read data
        raw_data = pandas.read_csv(fs[0], sep=self.dataset_info['sep'], 
                                   encoding=self.dataset_info['encoding'],
                                   names=['latitude', 'longitude', 'occupancy', 'time'],
                                   parse_dates=['time'], date_parser=mydateparser,
                                   header=None)
        
        #add the ID as a column
        raw_data['user_id'] = ['abboip']*len(raw_data)
        
        # convert the datetime
        tdf_dataset = skmob.TrajDataFrame(raw_data, latitude='latitude', 
                                          longitude='longitude', 
                                          user_id='user_id', 
                                          datetime='time')
        

        return tdf_dataset

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg83::bs4.BeautifulSoup+io.StringIO+pandas.DataFrame
# name: bs4_io_primitive
# summary: Uses bs4.BeautifulSoup, io.StringIO, pandas.DataFrame, pandas.concat across 4 repos
# anchor_symbols: ['bs4.BeautifulSoup', 'io.StringIO', 'pandas.DataFrame', 'pandas.concat', 'pandas.read_html', 'pandas.to_numeric', 'requests.get']
# observed in 4 repos: ['akfamily__akshare', 'devAmoghS__Machine-Learning-with-Python', 'justinzm__gopup', 'shashankvemuri__Finance']...

# --- from justinzm__gopup::gopup/index/index_weibo.py::_get_items ---
def _get_items(word="股票"):
    url = "https://data.weibo.com/index/ajax/newindex/searchword"
    payload = {"word": word}
    res = requests.post(url, data=payload, headers=index_weibo_headers)
    return {word: re.findall(r"\d+", res.json()["html"])[0]}

# --- from shashankvemuri__Finance::find_stocks/twitter_screener.py::scrape_most_active_stocks ---
def scrape_most_active_stocks():
    url = 'https://finance.yahoo.com/most-active/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    df = pd.read_html(str(soup), attrs={'class': 'W(100%)'})[0]
    df = df.drop(columns=['52 Week High'])
    return df

# --- from shashankvemuri__Finance::stock_data/send_top_movers.py::scrape_top_winners ---
def scrape_top_winners():
    url = 'https://finance.yahoo.com/gainers/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Parse the HTML content and extract data into a DataFrame
    df = pd.read_html(str(soup), attrs={'class': 'W(100%)'})[0]
    # Drop irrelevant columns for simplicity
    df = df.drop(columns=['52 Week High'])
    return df

# --- from devAmoghS__Machine-Learning-with-Python::natural_language_processing/utils.py::get_document ---
def get_document():
    url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html5lib')

    content = soup.find("div", "article-body")  # find article-body div
    regex = r"[\w']+|[\.]"  # matches a word or a period

    document = []

    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    return document

# --- from akfamily__akshare::akshare/futures/futures_comm_ctp.py::futures_fees_info ---
def futures_fees_info() -> pd.DataFrame:
    """
    openctp 期货交易费用参照表
    http://openctp.cn/fees.html
    :return: 期货交易费用参照表
    :rtype: pandas.DataFrame
    """
    url = "http://openctp.cn/fees.html"
    r = requests.get(url)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, features="lxml")
    datetime_str = soup.find("p").string.strip("Generated at ").strip(".")
    datetime_raw = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    temp_df = pd.read_html(StringIO(r.text))[0]
    temp_df["更新时间"] = datetime_raw.strftime("%Y-%m-%d %H:%M:%S")
    return temp_df

# --- from akfamily__akshare::akshare/option/option_comm_qihuo.py::option_comm_symbol ---
def option_comm_symbol() -> pd.DataFrame:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    url = "https://www.9qihuo.com/qiquanshouxufei"
    r = requests.get(url, verify=False)
    soup = BeautifulSoup(r.text, features="lxml")
    name = [
        item.string.strip()
        for item in soup.find(name="div", attrs={"id": "inst_list"}).find_all(name="a")
    ]
    code = [
        item["href"].split("?")[1].split("=")[1]
        for item in soup.find(name="div", attrs={"id": "inst_list"}).find_all(name="a")
    ]
    temp_df = pd.DataFrame([name, code]).T
    temp_df.columns = ["品种名称", "品种代码"]
    return temp_df

# --- from justinzm__gopup::gopup/life/charity.py::charity_organization ---
def charity_organization():
    """
    慈善中国-慈善组织查询
    http://cishan.chinanpo.gov.cn/biz/ma/csmh/a/csmhaindex.html
    :return: 慈善中国-慈善组织查询
    :rtype: pandas.DataFrame
    """
    page_num = _get_page_num_charity_organization()
    url = "http://cishan.chinanpo.gov.cn/biz/ma/csmh/a/csmhaDoSort.html"
    params = {
        "field": "aaex0131",
        "sort": "desc",
        "flag": "0",
    }
    outer_df = pd.DataFrame()
    for page in tqdm(range(1, page_num+1)):
        # page = 1
        params["pageNo"] = str(page)

        r = requests.post(url, params=params)
        inner_df = pd.read_html(r.text)[0]
        outer_df = outer_df.append(inner_df, ignore_index=True)
    return outer_df

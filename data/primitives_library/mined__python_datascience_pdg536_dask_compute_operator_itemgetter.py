# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg536::dask.compute+operator.itemgetter
# name: dask_operator_primitive
# summary: Uses dask.compute, operator.itemgetter across 2 repos
# anchor_symbols: ['dask.compute', 'operator.itemgetter']
# observed in 2 repos: ['encord-team__encord-active', 'sfu-db__dataprep']...

# --- from encord-team__encord-active::src/encord_active/lib/common/data_utils.py::url_to_file_path ---
def url_to_file_path(url: str, project_dir: Path) -> Optional[Path]:
    if url.startswith(("/", "./", "../", "~/")):
        return Path(url)
    if url.startswith("file:"):
        return Path(unquote(urlparse(url).path))
    if url.startswith("relative://"):
        relative_path = url[len("relative://") :]
        return project_dir / Path(relative_path)
    if url.startswith("absolute://"):
        absolute_path = url[len("absolute:/") :]
        return Path(absolute_path)
    return None

# --- from sfu-db__dataprep::dataprep/clean/clean_url.py::_format_url ---
def _format_url(
    url: Any, col: str, remove_auth: Union[bool, List[str]], split: bool, errors: str
) -> Any:
    """
    This function formats the input value "url"

    The last two components of the returned tuple hold the following codes:
        the first component: code indicating how the value was transformed (see below)
        the second component: the count of auth queries removed, if applicable
    In the first component, there are the following four codes:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value was parsed and DID NOT have authentication queries removed
        3 := the value was parsed and DID have authentication querires removed
    """
    # pylint: disable=too-many-locals

    # check if the url is a valid URL, returns a "status" value "null" (url is null),
    # "unknwon" (url is not a URL), and "success" (url is a URL)
    status = _check_url(url, True)

    if status == "null":
        return (np.nan, np.nan, np.nan, np.nan, 0, 0) if split else (np.nan, 0, 0)
    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"Unable to parse value {url}")
        result = url if errors == "ignore" else np.nan
        return (result, np.nan, np.nan, np.nan, 1, 0) if split else (result, 1, 0)

    # regex for finding the query / params and values
    re_queries = re.findall(QUERY_REGEX, url)
    all_queries = dict((y, z) for _, y, z in re_queries)

    # initialize the removed authentication code and count for the stats
    rem_auth_code, rem_auth_cnt = 2, 0
    # removing auth queries
    if remove_auth:
        to_remove = AUTH_VALUES if isinstance(remove_auth, bool) else UNIFIED_AUTH_LIST
        filtered_queries = {k: v for k, v in all_queries.items() if k not in to_remove}

        # count of removed auth queries
        rem_auth_cnt = len(all_queries) - len(filtered_queries)
        # code to indicate whether queries were removed
        rem_auth_code = 2 if rem_auth_cnt == 0 else 3

    # parse the url using urllib
    parsed = urlparse(url)

    # extracting params
    scheme = parsed.scheme
    host = parsed.hostname if parsed.hostname else ""
    path = parsed.path if parsed.path else ""
    cleaned_url = unquote(f"{scheme}://{host}{path}").replace(" ", "")
    queries = filtered_queries if remove_auth else all_queries

    # returning the type based upon the split parameter.
    if split:
        return scheme, host, cleaned_url, queries, rem_auth_code, rem_auth_cnt
    return (
        {"scheme": scheme, "host": host, f"{col}_clean": cleaned_url, "queries": queries},
        rem_auth_code,
        rem_auth_cnt,
    )

# --- from sfu-db__dataprep::dataprep/clean/clean_au_abn.py::clean_au_abn ---
def clean_au_abn(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Australian Business Numbers (ABNs) type data in a DataFrame column.

    Parameters
    ----------
        df
            A pandas or Dask DataFrame containing the data to be cleaned.
        col
            The name of the column containing data of ABN type.
        output_format
            The output format of standardized number string.
            If output_format = 'compact', return string without any separators or whitespace.
            If output_format = 'standard', return string with proper separators and whitespace.

            (default: "standard")
        inplace
           If True, delete the column containing the data that was cleaned.
           Otherwise, keep the original column.

           (default: False)
        errors
            How to handle parsing errors.
            - ‘coerce’: invalid parsing will be set to NaN.
            - ‘ignore’: invalid parsing will return the input.
            - ‘raise’: invalid parsing will raise an exception.

            (default: 'coerce')
        progress
            If True, display a progress bar.

            (default: True)

    Examples
    --------
    Clean a column of ABN data.

    >>> df = pd.DataFrame({
            "abn": [
            "51824753556",
            "99999999999",]
            })
    >>> clean_au_abn(df, 'abn')
            abn                 abn_clean
    0       51824753556         51 824 753 556
    1       99999999999         NaN
    """

    if output_format not in {"compact", "standard"}:
        raise ValueError(
            f"output_format {output_format} is invalid. " 'It needs to be "compact" or "standard".'
        )

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format(x, output_format, errors) for x in srs],
        meta=object,
    )

    df = df.assign(
        _temp_=df["clean_code_tup"].map(itemgetter(0)),
    )

    df = df.rename(columns={"_temp_": f"{column}_clean"})

    df = df.drop(columns=["clean_code_tup"])

    if inplace:
        df[column] = df[f"{column}_clean"]
        df = df.drop(columns=f"{column}_clean")
        df = df.rename(columns={column: f"{column}_clean"})

    with ProgressBar(minimum=1, disable=not progress):
        df = dask.compute()

    return df

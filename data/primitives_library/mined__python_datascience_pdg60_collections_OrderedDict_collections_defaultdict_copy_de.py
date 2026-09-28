# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg60::collections.OrderedDict+collections.defaultdict+copy.deepcopy
# name: collections_copy_primitive
# summary: Uses collections.OrderedDict, collections.defaultdict, copy.deepcopy, databricks.koalas.CategoricalIndex across 13 repos
# anchor_symbols: ['collections.OrderedDict', 'collections.defaultdict', 'copy.deepcopy', 'databricks.koalas.CategoricalIndex', 'databricks.koalas.DataFrame', 'databricks.koalas.DatetimeIndex']
# observed in 13 repos: ['AlexIoannides__pyspark-example-project', 'Data-Centric-AI-Community__fg-data-profiling', 'alteryx__evalml', 'capitalone__datacompy', 'databricks__koalas']...

# --- from databricks__koalas::databricks/koalas/tests/indexes/test_base.py::IndexesTest.test_index_nlevels ---
def test_index_nlevels(self):
        pdf = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index(["a", "b", "c"]))
        kdf = ks.from_pandas(pdf)

        self.assertEqual(kdf.index.nlevels, 1)

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/utils/common.py::convert_timestamp_to_datetime ---
def convert_timestamp_to_datetime(timestamp: int) -> datetime:
    if timestamp >= 0:
        return datetime.fromtimestamp(timestamp)
    else:
        return datetime(1970, 1, 1) + timedelta(seconds=int(timestamp))

# --- from databricks__koalas::databricks/koalas/tests/indexes/test_base.py::IndexesTest.test_dropna ---
def test_dropna(self):
        pidx = pd.Index([np.nan, 2, 4, 1, np.nan, 3])
        kidx = ks.from_pandas(pidx)

        self.assert_eq(kidx.dropna(), pidx.dropna())
        self.assert_eq((kidx + 1).dropna(), (pidx + 1).dropna())

# --- from xorbitsai__xorbits::python/xorbits/_mars/dataframe/base/_duplicate.py::validate_subset ---
def validate_subset(df, subset):
    if subset is None:
        return subset
    if not is_list_like(subset):
        subset = [subset]
    else:
        subset = list(subset)

    for s in subset:
        if s not in df.dtypes:
            raise KeyError(pd.Index([s]))

    return subset

# --- from capitalone__datacompy::tests/test_polars.py::test_sensitive_columns_hide_hide ---
def test_sensitive_columns_hide_hide():
    df1 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    df2 = pl.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 0}])
    compare = PolarsCompare(df1, df2, join_columns=["a"])
    compare.hide_sensitive_columns(["b"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "sensitive columns are already hidden, call reveal_sensitive_columns() first"
        ),
    ):
        compare.hide_sensitive_columns(["c"])

# --- from recommenders-team__recommenders::recommenders/datasets/criteo.py::get_spark_schema ---
def get_spark_schema(header=DEFAULT_HEADER):
    """Get Spark schema from header.

    Args:
        header (list): Dataset header names.

    Returns:
        pyspark.sql.types.StructType: Spark schema.
    """
    # create schema
    schema = StructType()
    # do label + ints
    n_ints = 14
    for i in range(n_ints):
        schema.add(StructField(header[i], IntegerType()))
    # do categoricals
    for i in range(26):
        schema.add(StructField(header[i + n_ints], StringType()))
    return schema

# --- from Data-Centric-AI-Community__fg-data-profiling::tests/backends/spark_backend/test_sample_spark.py::df_empty ---
def df_empty(spark_session):
    data_pandas = pd.DataFrame({"make": [], "registration": [], "year": []})
    # Turn the data into a Spark DataFrame, self.spark comes from our PySparkTest base class
    schema = StructType(
        {
            StructField("make", StringType(), True),
            StructField("registration", StringType(), True),
            StructField("year", IntegerType(), True),
        }
    )
    data_spark = spark_session.createDataFrame(data_pandas, schema=schema)
    return data_spark

# --- from stitchfix__hamilton::tests/test_default_data_quality.py::test_to_ensure_all_validators_added_to_default_validator_list ---
def test_to_ensure_all_validators_added_to_default_validator_list():
    def predicate(maybe_cls: Any) -> bool:
        if not inspect.isclass(maybe_cls):
            return False
        return issubclass(maybe_cls, BaseDefaultValidator) and maybe_cls != BaseDefaultValidator

    all_subclasses = inspect.getmembers(default_validators, predicate)
    missing_classes = [
        item
        for (_, item) in all_subclasses
        if item not in default_validators.AVAILABLE_DEFAULT_VALIDATORS
    ]
    assert len(missing_classes) == 0

# --- from AlexIoannides__pyspark-example-project::jobs/etl_job.py::transform_data ---
def transform_data(df, steps_per_floor_):
    """Transform original dataset.

    :param df: Input DataFrame.
    :param steps_per_floor_: The number of steps per-floor at 43 Tanner
        Street.
    :return: Transformed DataFrame.
    """
    df_transformed = (
        df
        .select(
            col('id'),
            concat_ws(
                ' ',
                col('first_name'),
                col('second_name')).alias('name'),
               (col('floor') * lit(steps_per_floor_)).alias('steps_to_desk')))

    return df_transformed

# --- from rpy2__rpy2::rpy2-robjects/src/rpy2/robjects/vectors.py::POSIXct._datetime_from_timestamp ---
def _datetime_from_timestamp(ts, tz) -> datetime.datetime:
        """Platform-dependent conversion from timestamp to datetime"""
        if os.name != 'nt' or ts > 0:
            return datetime.datetime.fromtimestamp(ts, tz)
        else:
            dt_utc = (datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) +
                      datetime.timedelta(seconds=ts))
            dt = dt_utc.replace(tzinfo=tz)
            offset = dt.utcoffset()
            if offset is None:
                return dt
            else:
                return dt + offset

# --- from capitalone__datacompy::tests/test_spark.py::test_custom_comparator_spark.StringLengthComparator.compare ---
def compare(self, dataframe, col1, col2):
            base_dtype, compare_dtype = get_spark_column_dtypes(dataframe, col1, col2)
            base_string_type = any(
                base_dtype.startswith(t) for t in PYSPARK_STRING_TYPE
            )
            compare_string_type = any(
                compare_dtype.startswith(t) for t in PYSPARK_STRING_TYPE
            )
            if base_string_type and compare_string_type:
                return when(
                    length(col(col1)) == length(col(col2)), lit(True)
                ).otherwise(lit(False))
            return None

# --- from rasbt__mlxtend::docs/make_api.py::get_functions_and_classes ---
def get_functions_and_classes(package):
    """Retun lists of functions and classes from a package.

    Parameters
    ----------
    package : Python package object

    Returns
    --------
    list, list : list of classes and functions
        Each sublist consists of [name, member] sublists.

    """
    classes, functions = [], []
    for name, member in inspect.getmembers(package):
        if not name.startswith("_"):
            if inspect.isclass(member):
                classes.append([name, member])
            elif inspect.isfunction(member):
                functions.append([name, member])
    return classes, functions

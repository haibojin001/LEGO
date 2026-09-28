# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg266::pandas.core.dtypes.common.is_datetime64_any_dtype+pandas.core.dtypes.common.is_numeric_dtype+pandas.to_numeric
# name: pandas_primitive
# summary: Uses pandas.core.dtypes.common.is_datetime64_any_dtype, pandas.core.dtypes.common.is_numeric_dtype, pandas.to_numeric across 3 repos
# anchor_symbols: ['pandas.core.dtypes.common.is_datetime64_any_dtype', 'pandas.core.dtypes.common.is_numeric_dtype', 'pandas.to_numeric']
# observed in 3 repos: ['deepchecks__deepchecks', 'feature-engine__feature_engine', 'modin-project__modin']...

# --- from feature-engine__feature_engine::feature_engine/variable_handling/_variable_type_checks.py::_is_convertible_to_num ---
def _is_convertible_to_num(column: pd.Series) -> bool:
    try:
        ser = pd.to_numeric(column)
    except (ValueError, TypeError):
        ser = column
    return is_numeric(ser)

# --- from deepchecks__deepchecks::deepchecks/utils/strings.py::is_string_column ---
def is_string_column(column: pd.Series) -> bool:
    """Determine whether a pandas series is string type."""
    if is_numeric_dtype(column):
        return False
    try:
        pd.to_numeric(column)
        return False
    except ValueError:
        return True
    # Non string objects like pd.Timestamp results in TypeError
    except TypeError:
        return False

# --- from deepchecks__deepchecks::deepchecks/utils/type_inference.py::get_column_type ---
def get_column_type(column: pd.Series) -> Literal['float', 'int', 'string', 'time', 'other']:
    """Get the type of column."""
    if is_float_dtype(column):
        return 'float'
    elif is_numeric_dtype(column):
        return 'int'
    elif is_datetime64_any_dtype(column):
        return 'time'

    try:
        column: pd.Series = pd.to_numeric(column)
        if is_float_dtype(column):
            return 'float'
        else:
            return 'int'
    except ValueError:
        return 'string'
    # Non-string objects like pd.Timestamp results in TypeError
    except TypeError:
        return 'other'

# --- from modin-project__modin::modin/pandas/groupby.py::DataFrameGroupBy.diff ---
def diff(self, periods=1, axis=lib.no_default):
        from .dataframe import DataFrame

        if axis is not lib.no_default:
            axis = self._df._get_axis_number(axis)
            self._deprecate_axis(axis, "diff")
        else:
            axis = 0

        # Should check for API level errors
        # Attempting to match pandas error behavior here
        if not isinstance(periods, int):
            raise TypeError(f"periods must be an int. got {type(periods)} instead")

        if isinstance(self._df, Series):
            if not is_numeric_dtype(self._df.dtypes):
                raise TypeError(
                    f"unsupported operand type for -: got {self._df.dtypes}"
                )
        elif isinstance(self._df, DataFrame) and axis == 0:
            for col, dtype in self._df.dtypes.items():
                # can't calculate diff on non-numeric columns, so check for non-numeric
                # columns that are not included in the `by`
                if not (
                    is_numeric_dtype(dtype) or is_datetime64_any_dtype(dtype)
                ) and not (
                    isinstance(self._by, BaseQueryCompiler) and col in self._by.columns
                ):
                    raise TypeError(f"unsupported operand type for -: got {dtype}")

        return self._wrap_aggregation(
            type(self._query_compiler).groupby_diff,
            agg_kwargs=dict(
                periods=periods,
                axis=axis,
            ),
        )

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg535::numpy.finfo+numpy.iinfo
# name: numpy_primitive
# summary: Uses numpy.finfo, numpy.iinfo across 4 repos
# anchor_symbols: ['numpy.finfo', 'numpy.iinfo']
# observed in 4 repos: ['DeepWisdom__AutoDL', 'firmai__pandasvault', 'sfu-db__dataprep', 'stitchfix__hamilton']...

# --- from firmai__pandasvault::pandasvault/__init__.py::reduce_mem_usage ---
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        gc.collect()
        if (col_type != object) and (str(col_type).lower() != 'category') and ('time' not in str(col_type).lower()):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                for int_type in (np.int8, np.int16, np.int32, np.int64):
                    if c_min > np.iinfo(int_type).min and c_max < np.iinfo(int_type).max:
                        df[col] = df[col].astype(int_type)
            else:
                for float_type in (np.float16, np.float32):
                    if c_min > np.finfo(float_type).min and c_max < np.finfo(float_type).max:
                        df[col] = df[col].astype(float_type)
                        break
                else:  # break is required with for/else
                    df[col] = df[col].astype(np.float64)
        else:
            if col_type == object:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# --- from sfu-db__dataprep::dataprep/clean/clean_df.py::_downcast_memory ---
def _downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to downcast the memory size of the DataFrame by using subtypes in
    numerical columns; for categorical types, downcast from `object` to `category`.
    """
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, j in enumerate(types):
        if "Int" in str(j) or "int" in str(j):
            if (
                df[cols[i]].min() > np.iinfo(np.int8).min
                and df[cols[i]].max() < np.iinfo(np.int8).max
            ):
                df[cols[i]] = df[cols[i]].astype("Int8")
            elif (
                df[cols[i]].min() > np.iinfo(np.int16).min
                and df[cols[i]].max() < np.iinfo(np.int16).max
            ):
                df[cols[i]] = df[cols[i]].astype("Int16")
            elif (
                df[cols[i]].min() > np.iinfo(np.int32).min
                and df[cols[i]].max() < np.iinfo(np.int32).max
            ):
                df[cols[i]] = df[cols[i]].astype("Int32")
            else:
                df[cols[i]] = df[cols[i]].astype("Int64")
        # Avoid forcing "float16" because it loses precision and is fragile
        elif "Float" in str(j) or "float" in str(j):
            if (
                df[cols[i]].min() > np.finfo(np.float16).min
                and df[cols[i]].max() < np.finfo(np.float32).max
            ):
                df[cols[i]] = pd.to_numeric(df[cols[i]], downcast="float")
            else:
                df[cols[i]] = df[cols[i]].astype("Float64")
        elif "Object" in str(j) or "object" in str(j):
            df[cols[i]] = df[cols[i]].astype("category")

    return df

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Tabular/utils/data_utils.py::downcast ---
def downcast(series, accuracy_loss=True, min_float_type='float16'):
    if series.dtype == np.int64:
        ii8 = np.iinfo(np.int8)
        ii16 = np.iinfo(np.int16)
        ii32 = np.iinfo(np.int32)
        max_value = series.max()
        min_value = series.min()
        
        if max_value <= ii8.max and min_value >= ii8.min:
            return series.astype(np.int8)
        elif max_value <= ii16.max and min_value >= ii16.min:
            return series.astype(np.int16)
        elif max_value <= ii32.max and min_value >= ii32.min:
            return series.astype(np.int32)
        else:
            return series
        
    elif series.dtype == np.float64:
        fi16 = np.finfo(np.float16)
        fi32 = np.finfo(np.float32)
        
        if accuracy_loss:
            max_value = series.max()
            min_value = series.min()
            if np.isnan(max_value):
                max_value = 0
            
            if np.isnan(min_value):
                min_value = 0
                
            if min_float_type=='float16' and max_value <= fi16.max and min_value >= fi16.min:
                return series.astype(np.float16)
            elif max_value <= fi32.max and min_value >= fi32.min:
                return series.astype(np.float32)
            else:
                return series
        else:
            tmp = series[~pd.isna(series)]
            if(len(tmp)==0):
                return series.astype(np.float16)
            
            if (tmp == tmp.astype(np.float16)).sum() == len(tmp):
                return series.astype(np.float16)
            elif (tmp == tmp.astype(np.float32)).sum() == len(tmp):
                return series.astype(np.float32)
           
            else:
                return series
            
    else:
        return series

# --- from stitchfix__hamilton::examples/model_examples/time-series/utils.py::reduce_mem_usage ---
def reduce_mem_usage(df: pd.DataFrame, name: str, verbose=True):
    """Taken from the notebook, this reduces the memory of each column if possible by downcasting the type if it can.

    :param df:
    :param name:
    :param verbose:
    :return:
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:

            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(
            "{}: Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                name, end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

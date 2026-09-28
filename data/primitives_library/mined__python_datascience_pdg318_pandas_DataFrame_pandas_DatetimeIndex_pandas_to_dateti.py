# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg318::pandas.DataFrame+pandas.DatetimeIndex+pandas.to_datetime
# name: pandas_primitive
# summary: Uses pandas.DataFrame, pandas.DatetimeIndex, pandas.to_datetime across 4 repos
# anchor_symbols: ['pandas.DataFrame', 'pandas.DatetimeIndex', 'pandas.to_datetime']
# observed in 4 repos: ['ScottfreeLLC__AlphaPy', 'alteryx__featuretools', 'lux-org__lux', 'meteostat__meteostat']...

# --- from lux-org__lux::tests/test_dates.py::test_period_to_altair ---
def test_period_to_altair(global_var):
    df = pd.read_csv("lux/data/car.csv")
    df["Year"] = pd.to_datetime(df["Year"], format="%Y")
    df["Year"] = pd.DatetimeIndex(df["Year"]).to_period(freq="A")
    from lux.vis.Vis import Vis

    vis = Vis(["Acceleration", "Horsepower", "Year=1972"], df)
    exported_code = vis.to_altair()

    assert "Year = 1972" in exported_code

# --- from lux-org__lux::tests/test_dates.py::test_period_filter ---
def test_period_filter(global_var):
    ldf = pd.read_csv("lux/data/car.csv")
    ldf["Year"] = pd.to_datetime(ldf["Year"], format="%Y")
    ldf["Year"] = pd.DatetimeIndex(ldf["Year"]).to_period(freq="A")

    from lux.vis.Vis import Vis

    vis = Vis(["Acceleration", "Horsepower", "Year=1972"], ldf)
    assert ldf.data_type["Year"] == "temporal"
    assert isinstance(vis._inferred_intent[2].value, str)

# --- from ScottfreeLLC__AlphaPy::alphapy/data.py::convert_data ---
def convert_data(df, index_column, intraday_data):
    r"""Convert the market data frame to canonical format.

    Parameters
    ----------
    df : pandas.DataFrame
        The intraday dataframe.
    index_column : str
        The name of the index column.
    intraday_data : bool
        Flag set to True if the frame contains intraday data.

    Returns
    -------
    df : pandas.DataFrame
        The canonical dataframe with date/time index.

    """

    # Standardize column names
    df = df.rename(columns = lambda x: x.lower().replace(' ',''))

    # Create the time/date index if not already done

    if not isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
        if intraday_data:
            dt_column = df['date'] + ' ' + df['time']
        else:
            dt_column = df['date']
        df[index_column] = pd.to_datetime(dt_column)
        df.set_index(pd.DatetimeIndex(df[index_column]),
                     drop=True, inplace=True)
        del df['date']
        if intraday_data:
            del df['time']

    # Make the remaining columns floating point

    cols_float = ['open', 'high', 'low', 'close', 'volume']
    df[cols_float] = df[cols_float].astype(float)

    # Order the frame by increasing date if necessary
    df = df.sort_index()

    return df

# --- from alteryx__featuretools::featuretools/demo/flight.py::_clean_data ---
def _clean_data(data):
    # Make column names snake case
    clean_data = data.rename(columns={col: convert(col) for col in data})

    # Chance crs -> "scheduled" and other minor clarifications
    clean_data = clean_data.rename(
        columns={
            "crs_arr_time": "scheduled_arr_time",
            "crs_dep_time": "scheduled_dep_time",
            "crs_elapsed_time": "scheduled_elapsed_time",
            "nas_delay": "national_airspace_delay",
            "origin_city_name": "origin_city",
            "dest_city_name": "dest_city",
            "cancelled": "canceled",
        },
    )

    # Combine strings like 0130 (1:30 AM) with dates (2017-01-01)
    clean_data["scheduled_dep_time"] = clean_data["scheduled_dep_time"].apply(
        lambda x: str(x),
    ) + clean_data["flight_date"].astype("str")

    # Parse combined string as a date
    clean_data.loc[:, "scheduled_dep_time"] = pd.to_datetime(
        clean_data["scheduled_dep_time"],
        format="%H%M%Y-%m-%d",
        errors="coerce",
    )

    clean_data["scheduled_elapsed_time"] = pd.to_timedelta(
        clean_data["scheduled_elapsed_time"],
        unit="m",
    )

    clean_data = _reconstruct_times(clean_data)

    # Create a time index 6 months before scheduled_dep
    clean_data.loc[:, "date_scheduled"] = pd.to_datetime(
        clean_data["scheduled_dep_time"],
    ).dt.date - pd.Timedelta("120d")

    # A null entry for a delay means no delay
    clean_data = _fill_labels(clean_data)

    # Nulls for scheduled values are too problematic. Remove them.
    clean_data = clean_data.dropna(
        axis="rows",
        subset=["scheduled_dep_time", "scheduled_arr_time"],
    )

    # Make a flight id. Define a flight as a combination of:
    # 1. carrier 2. flight number 3. origin airport 4. dest airport
    clean_data.loc[:, "flight_id"] = (
        clean_data["carrier"]
        + "-"
        + clean_data["flight_num"].apply(lambda x: str(x))
        + ":"
        + clean_data["origin"]
        + "->"
        + clean_data["dest"]
    )

    column_order = [
        "flight_id",
        "flight_num",
        "date_scheduled",
        "scheduled_dep_time",
        "scheduled_arr_time",
        "carrier",
        "origin",
        "origin_city",
        "origin_state",
        "dest",
        "dest_city",
        "dest_state",
        "distance_group",
        "dep_time",
        "arr_time",
        "dep_delay",
        "taxi_out",
        "taxi_in",
        "arr_delay",
        "diverted",
        "scheduled_elapsed_time",
        "air_time",
        "distance",
        "carrier_delay",
        "weather_delay",
        "national_airspace_delay",
        "security_delay",
        "late_aircraft_delay",
        "canceled",
    ]

    clean_data = clean_data[column_order]

    return clean_data

# --- from meteostat__meteostat::meteostat/providers/gsa/hourly.py::get_data ---
def get_data(
    station_id: str, parameters: list[str], start: datetime, end: datetime
) -> Optional[pd.DataFrame]:
    """
    Fetch data from GeoSphere Austria Data Hub API
    """
    logger.debug(
        f"Fetching hourly data for station '{station_id}' from {start} to {end}"
    )

    # Format dates as ISO 8601
    start_str = start.strftime("%Y-%m-%dT%H:%M")
    end_str = end.strftime("%Y-%m-%dT%H:%M")

    # Build URL
    url = f"{config.gsa_api_base_url}/station/historical/{RESOURCE_ID}"

    # Make request
    response = network_service.get(
        url,
        params={
            "parameters": ",".join(parameters),
            "station_ids": station_id,
            "start": start_str,
            "end": end_str,
            "output_format": "geojson",
        },
    )

    if response.status_code != 200:
        logger.warning(
            f"Failed to fetch data for station {station_id} (status: {response.status_code})"
        )
        return None

    try:
        data = response.json()

        if not data.get("features"):
            logger.info(f"No data returned for station {station_id}")
            return None

        # Get timestamps array
        timestamps = data.get("timestamps")
        if not timestamps:
            logger.warning("No timestamps in hourly response")
            return None

        # Extract time series data from GeoJSON response
        # New API format has timestamps at top level and data as arrays
        feature = data["features"][0]
        props = feature.get("properties", {})
        params_data = props.get("parameters", {})

        if not params_data:
            logger.info(f"No parameter data returned for station {station_id}")
            return None

        # Build DataFrame from timestamps and parameter arrays
        df_dict = {}
        for param in parameters:
            if param in params_data:
                param_info = params_data[param]
                if "data" in param_info:
                    df_dict[param] = param_info["data"]

        if not df_dict:
            return None

        # Create DataFrame with timestamps as index
        df = pd.DataFrame(df_dict)
        dt_index = pd.DatetimeIndex(pd.to_datetime(timestamps))
        df.index = dt_index.tz_localize(None)
        df.index.name = "time"

        # Sort by time
        df = df.sort_index()

        # Rename columns to Meteostat parameter names
        rename_map = {}
        for gsadh_param, meteostat_param in PARAMETER_MAPPING.items():
            if gsadh_param in df.columns:
                rename_map[gsadh_param] = meteostat_param

        df = df.rename(columns=rename_map)

        # Convert units where necessary
        if Parameter.WSPD in df.columns:
            df[Parameter.WSPD] = df[Parameter.WSPD].apply(ms_to_kmh)

        if Parameter.WPGT in df.columns:
            df[Parameter.WPGT] = df[Parameter.WPGT].apply(ms_to_kmh)

        if Parameter.TSUN in df.columns:
            df[Parameter.TSUN] = df[Parameter.TSUN].apply(hours_to_minutes)

        # Round values
        df = df.round(1)

        return df

    except Exception as error:
        logger.warning(f"Error parsing response: {error}", exc_info=True)
        return None

# --- from meteostat__meteostat::meteostat/providers/gsa/synop.py::get_data ---
def get_data(
    station_id: str,
    elevation: int | None,
    parameters: list[str],
    start: datetime,
    end: datetime,
) -> Optional[pd.DataFrame]:
    """
    Fetch SYNOP data from GeoSphere Austria Data Hub API
    """
    logger.debug(
        f"Fetching SYNOP hourly data for station '{station_id}' from {start} to {end}"
    )

    # Format dates as ISO 8601
    start_str = start.strftime("%Y-%m-%dT%H:%M")
    end_str = end.strftime("%Y-%m-%dT%H:%M")

    # Build URL
    url = f"{config.gsa_api_base_url}/station/historical/{RESOURCE_ID}"

    # Make request
    response = network_service.get(
        url,
        params={
            "parameters": ",".join(parameters),
            "station_ids": station_id,
            "start": start_str,
            "end": end_str,
            "output_format": "geojson",
        },
    )

    if response.status_code != 200:
        logger.warning(
            f"Failed to fetch SYNOP data for station {station_id} (status: {response.status_code})"
        )
        return None

    try:
        data = response.json()

        if not data.get("features"):
            logger.info(f"No SYNOP data returned for station {station_id}")
            return None

        # Get timestamps array
        timestamps = data.get("timestamps")
        if not timestamps:
            logger.warning("No timestamps in SYNOP response")
            return None

        # Extract time series data from GeoJSON response
        # New API format has timestamps at top level and data as arrays
        feature = data["features"][0]
        props = feature.get("properties", {})
        params_data = props.get("parameters", {})

        if not params_data:
            logger.info(f"No parameter data returned for station {station_id}")
            return None

        # Build DataFrame from timestamps and parameter arrays
        df_dict = {}
        for param in parameters:
            if param in params_data:
                param_info = params_data[param]
                if "data" in param_info:
                    df_dict[param] = param_info["data"]

        if not df_dict:
            return None

        # Create DataFrame with timestamps as index
        df = pd.DataFrame(df_dict)
        dt_index = pd.DatetimeIndex(pd.to_datetime(timestamps))
        df.index = dt_index.tz_localize(None)
        df.index.name = "time"

        # Sort by time
        df = df.sort_index()

        # Rename columns to Meteostat parameter names
        rename_map = {}
        for gsadh_param, meteostat_param in PARAMETER_MAPPING.items():
            if gsadh_param in df.columns:
                rename_map[gsadh_param] = meteostat_param

        df = df.rename(columns=rename_map)

        # Convert units where necessary
        if Parameter.WSPD in df.columns:
            df[Parameter.WSPD] = df[Parameter.WSPD].apply(ms_to_kmh)

        if Parameter.WPGT in df.columns:
            df[Parameter.WPGT] = df[Parameter.WPGT].apply(ms_to_kmh)

        # RRR returns -1 for no precipitation; convert to 0
        if Parameter.PRCP in df.columns:
            df[Parameter.PRCP] = df[Parameter.PRCP].replace(-1, 0)

        if Parameter.PRES in df.columns:
            df[Parameter.PRES] = df.apply(
                lambda row: pres_to_msl(row, elevation), axis=1
            )

        # Round values
        df = df.round(1)

        return df

    except Exception as error:
        logger.warning(f"Error parsing SYNOP response: {error}", exc_info=True)
        return None

# --- from ScottfreeLLC__AlphaPy::alphapy/data.py::get_market_data ---
def get_market_data(model, market_specs, group, lookback_period, intraday_data=False):
    r"""Get data from an external feed.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.
    market_specs : dict
        The specifications for controlling the MarketFlow pipeline.
    group : alphapy.Group
        The group of symbols.
    lookback_period : int
        The number of periods of data to retrieve.
    intraday_data : bool
        If True, then get intraday data.

    Returns
    -------
    n_periods : int
        The maximum number of periods actually retrieved.

    """

    # Unpack market specifications

    data_fractal = market_specs['data_fractal']
    subschema = market_specs['subschema']

    # Unpack model specifications

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']

    # Unpack group elements

    gspace = group.space
    schema = gspace.schema
    fractal = gspace.fractal

    # Determine the feed source

    if intraday_data:
        # intraday data (date and time)
        logger.info("%s Intraday Data [%s] for %d periods",
                    schema, data_fractal, lookback_period)
        index_column = 'datetime'
    else:
        # daily data or higher (date only)
        logger.info("%s Daily Data [%s] for %d periods",
                    schema, data_fractal, lookback_period)
        index_column = 'date'

    # Get the data from the relevant feed

    data_dir = SSEP.join([directory, 'data'])
    n_periods = 0
    resample_data = True if fractal != data_fractal else False

    # Date Arithmetic

    to_date = pd.to_datetime('today')
    from_date = to_date - pd.to_timedelta(lookback_period, unit='d')
    to_date = to_date.strftime('%Y-%m-%d')
    from_date = from_date.strftime('%Y-%m-%d')

    # Get the data from the specified data feed

    df = pd.DataFrame()
    for symbol in group.members:
        logger.info("Getting %s data from %s to %s",
                    symbol.upper(), from_date, to_date)
        # Locate the data source
        if schema == 'data':
            # local intraday or daily
            dspace = Space(gspace.subject, gspace.schema, data_fractal)
            fname = frame_name(symbol.lower(), dspace)
            df = read_frame(data_dir, fname, extension, separator)
        elif schema in data_dispatch_table.keys():
            df = data_dispatch_table[schema](schema,
                                             subschema,
                                             symbol,
                                             intraday_data,
                                             data_fractal,
                                             from_date,
                                             to_date,
                                             lookback_period)
        else:
            logger.error("Unsupported Data Source: %s", schema)
        # Now that we have content, standardize the data
        if not df.empty:
            logger.info("Rows: %d [%s]", len(df), data_fractal)
            # convert data to canonical form
            df = convert_data(df, index_column, intraday_data)
            # resample data and forward fill any NA values
            if resample_data:
                df = df.resample(fractal).agg({'open'   : 'first',
                                               'high'   : 'max',
                                               'low'    : 'min',
                                               'close'  : 'last',
                                               'volume' : 'sum'})
                df.dropna(axis=0, how='any', inplace=True)
                logger.info("Rows after Resampling at %s: %d",
                            fractal, len(df))
            # add intraday columns if necessary
            if intraday_data:
                df = enhance_intraday_data(df)
            # allocate global Frame
            newf = Frame(symbol.lower(), gspace, df)
            if newf is None:
                logger.error("Could not allocate Frame for: %s", symbol.upper())
            # calculate maximum number of periods
            df_len = len(df)
            if df_len > n_periods:
                n_periods = df_len
        else:
            logger.info("No DataFrame for %s", symbol.upper())

    # The number of periods actually retrieved
    return n_periods

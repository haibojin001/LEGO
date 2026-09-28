# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg213::dataclasses.asdict+dataclasses.is_dataclass
# name: dataclasses_primitive
# summary: Uses dataclasses.asdict, dataclasses.is_dataclass across 3 repos
# anchor_symbols: ['dataclasses.asdict', 'dataclasses.is_dataclass']
# observed in 3 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'ruc-datalab__DeepAnalyze', 'sematic-ai__sematic']...

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/skyrl_train/utils/tracking.py::_transform_params_to_json_serializable ---
def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(
        _transform_params_to_json_serializable,
        convert_list_to_dict=convert_list_to_dict,
    )

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {
                f"{i}": _transform(v) for i, v in enumerate(x)
            }
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x

# --- from sematic-ai__sematic::sematic/db/models/mixins/json_encodable_mixin.py::_to_json_encodable ---
def _to_json_encodable(value: Any, column: Column) -> Any:
    info = column.info

    if isinstance(value, datetime.datetime):
        # SQLite does not store timezone
        utc_value = datetime.datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            tzinfo=datetime.timezone.utc,
        )
        return utc_value.isoformat()

    if isinstance(value, enum.Enum):
        return value.value

    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)  # type: ignore

    if info.get(JSON_KEY, False) and value is not None:
        return json.loads(value)

    return value

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/profile_report.py::ProfileReport._render_json.encode_it ---
def encode_it(o: Any) -> Any:
            if is_dataclass(o):
                o = asdict(o)
            if isinstance(o, dict):
                return {encode_it(k): encode_it(v) for k, v in o.items()}
            else:
                if isinstance(o, (bool, int, float, str)):
                    return o
                elif isinstance(o, list):
                    return [encode_it(v) for v in o]
                elif isinstance(o, set):
                    return {encode_it(v) for v in o}
                elif isinstance(o, pd.Series):
                    return encode_it(o.to_list())
                elif isinstance(o, pd.DataFrame):
                    return encode_it(o.to_dict(orient="records"))
                elif isinstance(o, np.ndarray):
                    return encode_it(o.tolist())
                elif isinstance(o, Sample):
                    return encode_it(o.dict())
                elif isinstance(o, np.generic):
                    return o.item()
                else:
                    return str(o)

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/profile_report.py::ProfileReport._render_json ---
def _render_json(self) -> str:
        def encode_it(o: Any) -> Any:
            if is_dataclass(o):
                o = asdict(o)
            if isinstance(o, dict):
                return {encode_it(k): encode_it(v) for k, v in o.items()}
            else:
                if isinstance(o, (bool, int, float, str)):
                    return o
                elif isinstance(o, list):
                    return [encode_it(v) for v in o]
                elif isinstance(o, set):
                    return {encode_it(v) for v in o}
                elif isinstance(o, pd.Series):
                    return encode_it(o.to_list())
                elif isinstance(o, pd.DataFrame):
                    return encode_it(o.to_dict(orient="records"))
                elif isinstance(o, np.ndarray):
                    return encode_it(o.tolist())
                elif isinstance(o, Sample):
                    return encode_it(o.dict())
                elif isinstance(o, np.generic):
                    return o.item()
                else:
                    return str(o)

        description = self.description_set

        with tqdm(
            total=1, desc="Render JSON", disable=not self.config.progress_bar
        ) as pbar:
            description_dict = format_summary(description)
            description_dict = encode_it(description_dict)
            description_dict = redact_summary(description_dict, self.config)

            data = json.dumps(description_dict, indent=4)
            pbar.update()
        return data

import os

_prefix = "LT_"


def _get_value_str(name, default_value):
    found = os.environ.get(name)
    if found is None:
        return default_value
    return found


def _get_value_int(name, default_value):
    try:
        return int(os.environ[name])
    except:
        return default_value


def _get_value_bool(name, default_value):
    found = os.environ.get(name)
    if found in ("FALSE", "False", "false", "0"):
        return False
    if found in ("TRUE", "True", "true", "1"):
        return True
    return default_value


def _get_value(name, default_value, value_type):
    environment_name = _prefix + name
    readers = {
        "str": _get_value_str,
        "int": _get_value_int,
        "bool": _get_value_bool,
    }
    reader = readers.get(value_type)
    if reader is None:
        return default_value
    return reader(environment_name, default_value)


_option_specs = (
    ("HOST", "127.0.0.1", "str"),
    ("PORT", 5000, "int"),
    ("CHAR_LIMIT", -1, "int"),
    ("REQ_LIMIT", -1, "int"),
    ("REQ_LIMIT_STORAGE", "memory://", "str"),
    ("HOURLY_REQ_LIMIT", -1, "int"),
    ("HOURLY_REQ_LIMIT_DECAY", 0, "int"),
    ("DAILY_REQ_LIMIT", -1, "int"),
    ("REQ_FLOOD_THRESHOLD", -1, "int"),
    ("REQ_TIME_COST", -1, "int"),
    ("BATCH_LIMIT", -1, "int"),
    ("DEBUG", False, "bool"),
    ("SSL", None, "bool"),
    ("FRONTEND_LANGUAGE_SOURCE", "auto", "str"),
    ("FRONTEND_LANGUAGE_TARGET", "locale", "str"),
    ("FRONTEND_LANGUAGE", "", "str"),
    ("FRONTEND_TIMEOUT", 500, "int"),
    ("FRONTEND_TITLE", "", "str"),
    ("API_KEYS", False, "bool"),
    ("API_KEYS_DB_PATH", "db/api_keys.db", "str"),
    ("API_KEYS_REMOTE", "", "str"),
    ("GET_API_KEY_LINK", "", "str"),
    ("REQUIRE_API_KEY_ORIGIN", "", "str"),
    ("REQUIRE_API_KEY_SECRET", False, "bool"),
    ("UNDER_ATTACK", False, "bool"),
    ("REQUIRE_API_KEY_FINGERPRINT", False, "bool"),
    ("HIDE_API", False, "bool"),
    ("SHARED_STORAGE", "memory://", "str"),
    ("SECONDARY", False, "bool"),
    ("LOAD_ONLY", None, "str"),
    ("ALTERNATIVES_LIMIT", -1, "int"),
    ("THREADS", 4, "int"),
    ("TRUST_FORWARDED_FOR", False, "bool"),
    ("SUGGESTIONS", False, "bool"),
    ("DISABLE_FILES_TRANSLATION", False, "bool"),
    ("DISABLE_WEB_UI", False, "bool"),
    ("UPDATE_MODELS", False, "bool"),
    ("FORCE_UPDATE_MODELS", False, "bool"),
    ("METRICS", False, "bool"),
    ("METRICS_AUTH_TOKEN", "", "str"),
    ("TRANSLATION_CACHE", "", "str"),
    ("URL_PREFIX", "", "str"),
)

_default_options_objects = [
    {"name": option_name, "default_value": default, "value_type": option_type}
    for option_name, default, option_type in _option_specs
]

DEFAULT_ARGUMENTS = {
    option["name"]: _get_value(**option)
    for option in _default_options_objects
}
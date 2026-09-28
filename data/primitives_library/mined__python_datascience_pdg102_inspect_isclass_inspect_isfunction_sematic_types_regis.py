# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg102::inspect.isclass+inspect.isfunction+sematic.types.registry.is_supported_type_annotation
# name: inspect_sematic_primitive
# summary: Uses inspect.isclass, inspect.isfunction, sematic.types.registry.is_supported_type_annotation across 2 repos
# anchor_symbols: ['inspect.isclass', 'inspect.isfunction', 'sematic.types.registry.is_supported_type_annotation']
# observed in 2 repos: ['piskvorky__gensim', 'sematic-ai__sematic']...

# --- from sematic-ai__sematic::sematic/types/serialization.py::_parameter_repr._is_scalar ---
def _is_scalar(v):
        # is not a type or a class or a function or a sequence or a mapping
        return (
            (isinstance(v, str) or not isinstance(v, (typing.Sequence, typing.Mapping)))
            and not is_supported_type_annotation(v)
            and not inspect.isclass(v)
            and not inspect.isfunction(v)
        )

# --- from sematic-ai__sematic::sematic/types/serialization.py::_parameter_repr ---
def _parameter_repr(value: typing.Any) -> typing.Any:
    if is_supported_type_annotation(value):
        return {"type": _type_repr(value)}

    def _is_scalar(v):
        # is not a type or a class or a function or a sequence or a mapping
        return (
            (isinstance(v, str) or not isinstance(v, (typing.Sequence, typing.Mapping)))
            and not is_supported_type_annotation(v)
            and not inspect.isclass(v)
            and not inspect.isfunction(v)
        )

    if isinstance(value, typing.Sequence):
        if any(not _is_scalar(item) for item in value):
            return list(map(_parameter_repr, value))

    if isinstance(value, typing.Mapping):
        if any(not _is_scalar(item) for item in value.values()):
            return {k: _parameter_repr(v) for k, v in value.items()}

    return {"value": value}

# --- from piskvorky__gensim::gensim/utils.py::deprecated ---
def deprecated(reason):
    """Decorator to mark functions as deprecated.

    Calling a decorated function will result in a warning being emitted, using warnings.warn.
    Adapted from https://stackoverflow.com/a/40301488/8001386.

    Parameters
    ----------
    reason : str
        Reason of deprecation.

    Returns
    -------
    function
        Decorated function

    """
    if isinstance(reason, str):
        def decorator(func):
            fmt = "Call to deprecated `{name}` ({reason})."

            @wraps(func)
            def new_func1(*args, **kwargs):
                warnings.warn(
                    fmt.format(name=func.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                return func(*args, **kwargs)

            return new_func1
        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func = reason
        fmt = "Call to deprecated `{name}`."

        @wraps(func)
        def new_func2(*args, **kwargs):
            warnings.warn(
                fmt.format(name=func.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return new_func2

    else:
        raise TypeError(repr(type(reason)))

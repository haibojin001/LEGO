# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg705::base64.b64encode+pickle.dumps
# name: base64_pickle_primitive
# summary: Uses base64.b64encode, pickle.dumps across 2 repos
# anchor_symbols: ['base64.b64encode', 'pickle.dumps']
# observed in 2 repos: ['HDI-Project__ATM', 'run-house__kubetorch']...

# --- from HDI-Project__ATM::atm/utilities.py::object_to_base_64 ---
def object_to_base_64(obj):
    """ Pickle and base64-encode an object. """
    pickled = pickle.dumps(obj)
    return base64.b64encode(pickled)

# --- from run-house__kubetorch::python_client/kubetorch/serving/utils.py::_serialize_body ---
def _serialize_body(body: dict, serialization: str):
    if body is None:
        return {}

    # We only serialize args and kwargs, other settings like "workers" and "restart_procs" are needed inside
    # the http_server, outside the serialization boundary (e.g. the distributed processes)
    # We break them out here as separate params
    body = body or {}

    for kwarg in MAGIC_CALL_KWARGS:
        if kwarg in body.get("kwargs", {}):
            body[kwarg] = body["kwargs"].pop(kwarg)

    if serialization == "pickle":
        args_data = {"args": body.pop("args"), "kwargs": body.pop("kwargs")}
        pickled_args = pickle.dumps(args_data or {})
        encoded_args = base64.b64encode(pickled_args).decode("utf-8")
        body["data"] = encoded_args
        return body
    return body or {}

# --- from run-house__kubetorch::python_client/kubetorch/serving/http_server.py::_serialize_result ---
def _serialize_result(result, serialization: str):
    """Serialize the result based on the format."""
    if serialization == "pickle":
        try:
            pickled_result = pickle.dumps(result)
            encoded_result = base64.b64encode(pickled_result).decode("utf-8")
            return {"data": encoded_result}
        except Exception as e:
            logger.error(f"Failed to pickle result: {str(e)}")
            raise SerializationError(f"Result could not be serialized with pickle: {str(e)}")
    elif serialization == "json":
        # Default JSON serialization
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Result is not JSON serializable: {str(e)}")
            raise SerializationError(f"Result could not be serialized to JSON: {str(e)}")
    return result

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg51::numpy.float64+numpy.int32
# name: numpy_primitive
# summary: Uses numpy.float64, numpy.int32 across 2 repos
# anchor_symbols: ['numpy.float64', 'numpy.int32']
# observed in 2 repos: ['piskvorky__gensim', 'sinaptik-ai__pandas-ai']...

# --- from sinaptik-ai__pandas-ai::tests/unit_tests/helpers/test_json_encoder.py::test_custom_json_encoder_numpy_types ---
def test_custom_json_encoder_numpy_types():
    # Arrange
    obj = {
        "integer": np.int32(123),
        "float": np.float64(1.23),
        "array": np.array([1, 2, 3]),
    }
    expected_json = '{"integer": 123, "float": 1.23, "array": [1, 2, 3]}'

    # Act
    result = json.dumps(obj, cls=CustomJsonEncoder)

    # Assert
    assert result == expected_json

# --- from piskvorky__gensim::gensim/models/_fasttext_bin.py::_conv_field_to_bytes ---
def _conv_field_to_bytes(field_value, field_type):
    """
    Auxiliary function that converts `field_value` to bytes based on request `field_type`,
    for saving to the binary file.

    Parameters
    ----------
    field_value: numerical
        contains arguments of the string and start/end indexes of the bad portion.

    field_type: str
        currently supported `field_types` are `i` for 32-bit integer and `d` for 64-bit float
    """
    if field_type == 'i':
        return (np.int32(field_value).tobytes())
    elif field_type == 'd':
        return (np.float64(field_value).tobytes())
    else:
        raise NotImplementedError('Currently conversion to "%s" type is not implemmented.' % field_type)

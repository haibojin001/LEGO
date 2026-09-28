import math
import json

from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods


TYPES_MAP = {
    'bool': 'boolean',
    'list': 'array',
    'dict': 'object',
    'NoneType': 'null',
    'unicode': 'string',
    'str': 'string',
    'float': 'number',
    'int': 'number',
    'long': 'number',
    'OrderedDict': 'object',
    '_Projection': 'array',
    '_Expression': 'expref',
}

REVERSE_TYPES_MAP = {
    'boolean': ('bool',),
    'array': ('list', '_Projection'),
    'object': ('dict', 'OrderedDict'),
    'null': ('NoneType',),
    'string': ('unicode', 'str'),
    'number': ('float', 'int', 'long'),
    'expref': ('_Expression',),
}


def signature(*arguments):
    def decorator(function):
        function.signature = arguments
        return function
    return decorator


class FunctionRegistry(type):
    def __init__(cls, name, bases, attributes):
        cls._populate_function_table()
        super(FunctionRegistry, cls).__init__(name, bases, attributes)

    def _populate_function_table(cls):
        table = {}
        for method_name, method in get_methods(cls):
            if not method_name.startswith('_func_'):
                continue
            method_signature = getattr(method, 'signature', None)
            if method_signature is not None:
                table[method_name[6:]] = {
                    'function': method,
                    'signature': method_signature,
                }
        cls.FUNCTION_TABLE = table


class Functions(metaclass=FunctionRegistry):
    FUNCTION_TABLE = {}

    def call_function(self, function_name, resolved_args):
        try:
            function_spec = self.FUNCTION_TABLE[function_name]
        except KeyError:
            raise exceptions.UnknownFunctionError(
                'Unknown function: %s()' % function_name
            )
        self._validate_arguments(
            resolved_args,
            function_spec['signature'],
            function_name,
        )
        return function_spec['function'](self, *resolved_args)

    def _validate_arguments(self, args, expected_signature, function_name):
        if expected_signature and expected_signature[-1].get('variadic'):
            if len(args) < len(expected_signature):
                raise exceptions.VariadictArityError(
                    len(expected_signature), len(args), function_name
                )
        elif len(args) != len(expected_signature):
            raise exceptions.ArityError(
                len(expected_signature), len(args), function_name
            )
        self._type_check(args, expected_signature, function_name)

    def _type_check(self, actual, expected_signature, function_name):
        for index in range(len(expected_signature)):
            expected_types = expected_signature[index]['types']
            if expected_types:
                self._type_check_single(
                    actual[index],
                    expected_types,
                    function_name,
                )

    def _type_check_single(self, value, expected_types, function_name):
        allowed_types, allowed_subtypes = self._get_allowed_pytypes(
            expected_types
        )
        actual_type = type(value).__name__
        if actual_type not in allowed_types:
            raise exceptions.JMESPathTypeError(
                function_name,
                value,
                self._convert_to_jmespath_type(actual_type),
                expected_types,
            )
        if allowed_subtypes:
            self._subtype_check(
                value,
                allowed_subtypes,
                expected_types,
                function_name,
            )

    def _get_allowed_pytypes(self, types):
        top_level_types = []
        subtypes = []
        for type_name in types:
            pieces = type_name.split('-', 1)
            if len(pieces) == 2:
                type_name, subtype_name = pieces
                subtypes.append(REVERSE_TYPES_MAP[subtype_name])
            else:
                type_name = pieces[0]
            top_level_types.extend(REVERSE_TYPES_MAP[type_name])
        return top_level_types, subtypes

    def _subtype_check(self, value, allowed_subtypes, expected_types,
                       function_name):
        if len(allowed_subtypes) == 1:
            permitted = allowed_subtypes[0]
            for item in value:
                item_type = type(item).__name__
                if item_type not in permitted:
                    raise exceptions.JMESPathTypeError(
                        function_name,
                        item,
                        item_type,
                        expected_types,
                    )
        elif len(allowed_subtypes) > 1 and value:
            first_type = type(value[0]).__name__
            for permitted in allowed_subtypes:
                if first_type in permitted:
                    break
            else:
                raise exceptions.JMESPathTypeError(
                    function_name,
                    value[0],
                    first_type,
                    expected_types,
                )
            for item in value:
                item_type = type(item).__name__
                if item_type not in permitted:
                    raise exceptions.JMESPathTypeError(
                        function_name,
                        item,
                        item_type,
                        expected_types,
                    )

    def _convert_to_jmespath_type(self, pytype):
        return TYPES_MAP[pytype]

    def _create_key_func(self, expref, allowed_types, function_name):
        def key_function(value):
            result = expref.visit(expref.expression, value)
            result_type = self._convert_to_jmespath_type(
                type(result).__name__
            )
            if result_type not in allowed_types:
                raise exceptions.JMESPathTypeError(
                    function_name,
                    result,
                    result_type,
                    allowed_types,
                )
            return result
        return key_function

    @signature({'types': ['number']})
    def _func_abs(self, arg):
        return abs(arg)

    @signature({'types': ['array-number']})
    def _func_avg(self, arg):
        if arg:
            return sum(arg) / len(arg)
        return None

    @signature({'types': [], 'variadic': True})
    def _func_not_null(self, *arguments):
        for argument in arguments:
            if argument is not None:
                return argument

    @signature({'types': []})
    def _func_to_array(self, arg):
        if isinstance(arg, list):
            return arg
        return [arg]

    @signature({'types': []})
    def _func_to_string(self, arg):
        if isinstance(arg, STRING_TYPE):
            return arg
        return json.dumps(arg, separators=(',', ':'), default=str)

    @signature({'types': []})
    def _func_to_number(self, arg):
        if isinstance(arg, (list, dict, bool)):
            return None
        if arg is None:
            return None
        if isinstance(arg, (int, float)):
            return arg
        try:
            return int(arg)
        except ValueError:
            try:
                return float(arg)
            except ValueError:
                return None

    @signature({'types': ['array', 'string']}, {'types': []})
    def _func_contains(self, subject, search):
        return search in subject

    @signature({'types': ['string', 'array', 'object']})
    def _func_length(self, arg):
        return len(arg)

    @signature({'types': ['string']}, {'types': ['string']})
    def _func_ends_with(self, search, suffix):
        return search.endswith(suffix)

    @signature({'types': ['string']}, {'types': ['string']})
    def _func_starts_with(self, search, prefix):
        return search.startswith(prefix)

    @signature({'types': ['array', 'string']})
    def _func_reverse(self, arg):
        if isinstance(arg, STRING_TYPE):
            return arg[::-1]
        return list(reversed(arg))

    @signature({'types': ['number']})
    def _func_ceil(self, arg):
        return math.ceil(arg)

    @signature({'types': ['number']})
    def _func_floor(self, arg):
        return math.floor(arg)

    @signature({'types': ['string']}, {'types': ['array-string']})
    def _func_join(self, separator, array):
        return separator.join(array)

    @signature({'types': ['expref']}, {'types': ['array']})
    def _func_map(self, expref, arg):
        result = []
        for element in arg:
            result.append(expref.visit(expref.expression, element))
        return result

    @signature({'types': ['array-number', 'array-string']})
    def _func_max(self, arg):
        if arg:
            return max(arg)
        return None

    @signature({'types': ['object'], 'variadic': True})
    def _func_merge(self, *arguments):
        merged = {}
        for argument in arguments:
            merged.update(argument)
        return merged

    @signature({'types': ['array-number', 'array-string']})
    def _func_min(self, arg):
        if arg:
            return min(arg)
        return None

    @signature({'types': ['array-number']})
    def _func_sum(self, arg):
        return sum(arg)

    @signature({'types': ['array-string', 'array-number']})
    def _func_sort(self, arg):
        return list(sorted(arg))

    @signature({'types': ['object']})
    def _func_keys(self, arg):
        return list(arg.keys())

    @signature({'types': ['object']})
    def _func_values(self, arg):
        return list(arg.values())

    @signature({'types': []})
    def _func_type(self, arg):
        return TYPES_MAP[type(arg).__name__]

    @signature({'types': ['array']}, {'types': ['expref']})
    def _func_sort_by(self, arg, expref):
        if not arg:
            return arg
        required_type = self._convert_to_jmespath_type(
            type(expref.visit(expref.expression, arg[0])).__name__
        )
        if required_type not in ['number', 'string']:
            raise exceptions.JMESPathTypeError(
                'sort_by',
                arg[0],
                required_type,
                ['string', 'number'],
            )
        key_function = self._create_key_func(
            expref,
            [required_type],
            'sort_by',
        )
        return list(sorted(arg, key=key_function))

    @signature({'types': ['array']}, {'types': ['expref']})
    def _func_max_by(self, arg, expref):
        if arg:
            return max(
                arg,
                key=self._create_key_func(
                    expref,
                    ['number', 'string'],
                    'max_by',
                ),
            )
        return None

    @signature({'types': ['array']}, {'types': ['expref']})
    def _func_min_by(self, arg, expref):
        if arg:
            return min(
                arg,
                key=self._create_key_func(
                    expref,
                    ['number', 'string'],
                    'min_by',
                ),
            )
        return None
import operator
from numbers import Number

from jmespath import functions
from jmespath.compat import string_type


def _is_actual_number(value):
    if isinstance(value, bool):
        return False
    return isinstance(value, Number)


def _is_special_number_case(left, right):
    if _is_actual_number(left) and left in (0, 1):
        return isinstance(right, bool)
    if _is_actual_number(right) and right in (0, 1):
        return isinstance(left, bool)


def _equals(left, right):
    if _is_special_number_case(left, right):
        return False
    return left == right


def _is_comparable(value):
    return _is_actual_number(value) or isinstance(value, string_type)


class Options(object):
    """Options to control how a JMESPath function is evaluated."""

    def __init__(self, dict_cls=None, custom_functions=None):
        self.dict_cls = dict_cls
        self.custom_functions = custom_functions


class _Expression(object):
    def __init__(self, expression, interpreter):
        self.expression = expression
        self.interpreter = interpreter

    def visit(self, node, *args, **kwargs):
        return self.interpreter.visit(node, *args, **kwargs)


class Visitor(object):
    def __init__(self):
        self._method_cache = {}

    def visit(self, node, *args, **kwargs):
        node_type = node['type']
        method = self._method_cache.get(node_type)
        if method is None:
            method = getattr(self, 'visit_%s' % node_type, self.default_visit)
            self._method_cache[node_type] = method
        return method(node, *args, **kwargs)

    def default_visit(self, node, *args, **kwargs):
        raise NotImplementedError("default_visit")


class TreeInterpreter(Visitor):
    COMPARATOR_FUNC = {
        'eq': _equals,
        'ne': lambda left, right: not _equals(left, right),
        'lt': operator.lt,
        'gt': operator.gt,
        'lte': operator.le,
        'gte': operator.ge,
    }
    _EQUALITY_OPS = ['eq', 'ne']
    MAP_TYPE = dict

    def __init__(self, options=None):
        super(TreeInterpreter, self).__init__()
        self._dict_cls = self.MAP_TYPE
        if options is None:
            options = Options()
        self._options = options
        if options.dict_cls is not None:
            self._dict_cls = options.dict_cls
        if options.custom_functions is not None:
            self._functions = options.custom_functions
        else:
            self._functions = functions.Functions()

    def default_visit(self, node, *args, **kwargs):
        raise NotImplementedError(node['type'])

    def visit_subexpression(self, node, value):
        current = value
        for child in node['children']:
            current = self.visit(child, current)
        return current

    def visit_field(self, node, value):
        try:
            return value.get(node['value'])
        except AttributeError:
            return None

    def visit_comparator(self, node, value):
        operation = node['value']
        comparator = self.COMPARATOR_FUNC[operation]
        left = self.visit(node['children'][0], value)
        right = self.visit(node['children'][1], value)
        if operation in self._EQUALITY_OPS:
            return comparator(left, right)
        if not (_is_comparable(left) and _is_comparable(right)):
            return None
        return comparator(left, right)

    def visit_current(self, node, value):
        return value

    def visit_expref(self, node, value):
        return _Expression(node['children'][0], self)

    def visit_function_expression(self, node, value):
        arguments = []
        for child in node['children']:
            arguments.append(self.visit(child, value))
        return self._functions.call_function(node['value'], arguments)

    def visit_filter_projection(self, node, value):
        base = self.visit(node['children'][0], value)
        if not isinstance(base, list):
            return None
        projection = node['children'][1]
        predicate = node['children'][2]
        result = []
        for element in base:
            if self._is_true(self.visit(predicate, element)):
                projected = self.visit(projection, element)
                if projected is not None:
                    result.append(projected)
        return result

    def visit_flatten(self, node, value):
        base = self.visit(node['children'][0], value)
        if not isinstance(base, list):
            return None
        result = []
        for element in base:
            if isinstance(element, list):
                result.extend(element)
            else:
                result.append(element)
        return result

    def visit_identity(self, node, value):
        return value

    def visit_index(self, node, value):
        if not isinstance(value, list):
            return None
        try:
            return value[node['value']]
        except IndexError:
            return None

    def visit_index_expression(self, node, value):
        current = value
        for child in node['children']:
            current = self.visit(child, current)
        return current

    def visit_slice(self, node, value):
        if not isinstance(value, list):
            return None
        return value[slice(*node['children'])]

    def visit_key_val_pair(self, node, value):
        return self.visit(node['children'][0], value)

    def visit_literal(self, node, value):
        return node['value']

    def visit_multi_select_dict(self, node, value):
        if value is None:
            return None
        result = self._dict_cls()
        for child in node['children']:
            result[child['value']] = self.visit(child, value)
        return result

    def visit_multi_select_list(self, node, value):
        if value is None:
            return None
        return [self.visit(child, value) for child in node['children']]

    def visit_or_expression(self, node, value):
        result = self.visit(node['children'][0], value)
        if self._is_false(result):
            result = self.visit(node['children'][1], value)
        return result

    def visit_and_expression(self, node, value):
        result = self.visit(node['children'][0], value)
        if self._is_false(result):
            return result
        return self.visit(node['children'][1], value)

    def visit_not_expression(self, node, value):
        result = self.visit(node['children'][0], value)
        if _is_actual_number(result) and result == 0:
            return False
        return not result

    def visit_pipe(self, node, value):
        current = value
        for child in node['children']:
            current = self.visit(child, current)
        return current

    def visit_projection(self, node, value):
        base = self.visit(node['children'][0], value)
        if not isinstance(base, list):
            return None
        result = []
        for element in base:
            projected = self.visit(node['children'][1], element)
            if projected is not None:
                result.append(projected)
        return result

    def visit_value_projection(self, node, value):
        base = self.visit(node['children'][0], value)
        if base is None:
            return None
        if not isinstance(base, dict):
            return None
        result = []
        for element in base.values():
            projected = self.visit(node['children'][1], element)
            if projected is not None:
                result.append(projected)
        return result

    def _is_false(self, value):
        return (
            value == '' or
            value == [] or
            value == {} or
            value is None or
            value is False
        )

    def _is_true(self, value):
        return not self._is_false(value)
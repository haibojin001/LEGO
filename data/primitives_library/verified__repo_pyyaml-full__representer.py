__all__ = [
    'BaseRepresenter',
    'SafeRepresenter',
    'Representer',
    'RepresenterError',
]

import base64
import collections
import copyreg
import datetime
import types

from .error import *
from .nodes import *


class RepresenterError(YAMLError):
    pass


class BaseRepresenter:

    yaml_representers = {}
    yaml_multi_representers = {}

    def __init__(self, default_style=None, default_flow_style=False,
                 sort_keys=True):
        self.default_style = default_style
        self.default_flow_style = default_flow_style
        self.sort_keys = sort_keys
        self.represented_objects = {}
        self.object_keeper = []
        self.alias_key = None

    def represent(self, data):
        node = self.represent_data(data)
        self.serialize(node)
        self.represented_objects = {}
        self.object_keeper = []
        self.alias_key = None

    def represent_data(self, data):
        if self.ignore_aliases(data):
            self.alias_key = None
        else:
            self.alias_key = id(data)

        if self.alias_key is not None:
            if self.alias_key in self.represented_objects:
                return self.represented_objects[self.alias_key]
            self.object_keeper.append(data)

        data_types = type(data).__mro__

        direct_representer = self.yaml_representers.get(data_types[0])
        if direct_representer is not None:
            return direct_representer(self, data)

        for data_type in data_types:
            multi_representer = self.yaml_multi_representers.get(data_type)
            if multi_representer is not None:
                return multi_representer(self, data)

        fallback_multi = self.yaml_multi_representers.get(None)
        if fallback_multi is not None:
            return fallback_multi(self, data)

        fallback_direct = self.yaml_representers.get(None)
        if fallback_direct is not None:
            return fallback_direct(self, data)

        return ScalarNode(None, str(data))

    @classmethod
    def add_representer(cls, data_type, representer):
        if 'yaml_representers' not in cls.__dict__:
            cls.yaml_representers = cls.yaml_representers.copy()
        cls.yaml_representers[data_type] = representer

    @classmethod
    def add_multi_representer(cls, data_type, representer):
        if 'yaml_multi_representers' not in cls.__dict__:
            cls.yaml_multi_representers = cls.yaml_multi_representers.copy()
        cls.yaml_multi_representers[data_type] = representer

    def represent_scalar(self, tag, value, style=None):
        if style is None:
            style = self.default_style
        node = ScalarNode(tag, value, style=style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        return node

    def represent_sequence(self, tag, sequence, flow_style=None):
        items = []
        node = SequenceNode(tag, items, flow_style=flow_style)

        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node

        simple = True
        for item in sequence:
            child = self.represent_data(item)
            if not isinstance(child, ScalarNode) or child.style:
                simple = False
            items.append(child)

        if flow_style is None:
            if self.default_flow_style is None:
                node.flow_style = simple
            else:
                node.flow_style = self.default_flow_style

        return node

    def represent_mapping(self, tag, mapping, flow_style=None):
        pairs = []
        node = MappingNode(tag, pairs, flow_style=flow_style)

        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node

        simple = True

        if hasattr(mapping, 'items'):
            mapping = list(mapping.items())
            if self.sort_keys:
                try:
                    mapping = sorted(mapping)
                except TypeError:
                    pass

        for key, item in mapping:
            represented_key = self.represent_data(key)
            represented_item = self.represent_data(item)

            if not isinstance(represented_key, ScalarNode) or represented_key.style:
                simple = False
            if not isinstance(represented_item, ScalarNode) or represented_item.style:
                simple = False

            pairs.append((represented_key, represented_item))

        if flow_style is None:
            if self.default_flow_style is None:
                node.flow_style = simple
            else:
                node.flow_style = self.default_flow_style

        return node

    def ignore_aliases(self, data):
        return False


class SafeRepresenter(BaseRepresenter):

    def ignore_aliases(self, data):
        if data is None:
            return True
        if isinstance(data, tuple) and data == ():
            return True
        return isinstance(data, (str, bytes, bool, int, float))

    def represent_none(self, data):
        return self.represent_scalar('tag:yaml.org,2002:null', 'null')

    def represent_str(self, data):
        return self.represent_scalar('tag:yaml.org,2002:str', data)

    def represent_binary(self, data):
        encoder = getattr(base64, 'encodebytes', None)
        if encoder is None:
            encoded = base64.encodestring(data)
        else:
            encoded = encoder(data)
        return self.represent_scalar(
            'tag:yaml.org,2002:binary',
            encoded.decode('ascii'),
            style='|',
        )

    def represent_bool(self, data):
        value = 'true' if data else 'false'
        return self.represent_scalar('tag:yaml.org,2002:bool', value)

    def represent_int(self, data):
        return self.represent_scalar('tag:yaml.org,2002:int', str(data))

    inf_value = 1e300
    while repr(inf_value) != repr(inf_value * inf_value):
        inf_value *= inf_value

    def represent_float(self, data):
        if data != data or (data == 0.0 and data == 1.0):
            value = '.nan'
        elif data == self.inf_value:
            value = '.inf'
        elif data == -self.inf_value:
            value = '-.inf'
        else:
            value = repr(data).lower()
            if '.' not in value and 'e' in value:
                value = value.replace('e', '.0e', 1)

        return self.represent_scalar('tag:yaml.org,2002:float', value)

    def represent_list(self, data):
        return self.represent_sequence('tag:yaml.org,2002:seq', data)

    def represent_dict(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data)

    def represent_set(self, data):
        mapping = {}
        for key in data:
            mapping[key] = None
        return self.represent_mapping('tag:yaml.org,2002:set', mapping)

    def represent_date(self, data):
        return self.represent_scalar(
            'tag:yaml.org,2002:timestamp',
            data.isoformat(),
        )

    def represent_datetime(self, data):
        return self.represent_scalar(
            'tag:yaml.org,2002:timestamp',
            data.isoformat(' '),
        )

    def represent_yaml_object(self, tag, data, cls, flow_style=None):
        if hasattr(data, '__getstate__'):
            state = data.__getstate__()
        else:
            state = data.__dict__.copy()
        return self.represent_mapping(tag, state, flow_style=flow_style)

    def represent_undefined(self, data):
        raise RepresenterError("cannot represent an object", data)


SafeRepresenter.add_representer(type(None), SafeRepresenter.represent_none)
SafeRepresenter.add_representer(str, SafeRepresenter.represent_str)
SafeRepresenter.add_representer(bytes, SafeRepresenter.represent_binary)
SafeRepresenter.add_representer(bool, SafeRepresenter.represent_bool)
SafeRepresenter.add_representer(int, SafeRepresenter.represent_int)
SafeRepresenter.add_representer(float, SafeRepresenter.represent_float)
SafeRepresenter.add_representer(list, SafeRepresenter.represent_list)
SafeRepresenter.add_representer(tuple, SafeRepresenter.represent_list)
SafeRepresenter.add_representer(dict, SafeRepresenter.represent_dict)
SafeRepresenter.add_representer(set, SafeRepresenter.represent_set)
SafeRepresenter.add_representer(datetime.date, SafeRepresenter.represent_date)
SafeRepresenter.add_representer(
    datetime.datetime,
    SafeRepresenter.represent_datetime,
)
SafeRepresenter.add_representer(None, SafeRepresenter.represent_undefined)


class Representer(SafeRepresenter):

    def represent_complex(self, data):
        if data.imag == 0.0:
            value = '%r' % data.real
        elif data.real == 0.0:
            value = '%rj' % data.imag
        elif data.imag > 0:
            value = '%r+%rj' % (data.real, data.imag)
        else:
            value = '%r%rj' % (data.real, data.imag)
        return self.represent_scalar('tag:yaml.org,2002:python/complex', value)

    def represent_tuple(self, data):
        return self.represent_sequence('tag:yaml.org,2002:python/tuple', data)

    def represent_name(self, data):
        name = '%s.%s' % (data.__module__, data.__name__)
        return self.represent_scalar(
            'tag:yaml.org,2002:python/name:' + name,
            '',
        )

    def represent_module(self, data):
        return self.represent_scalar(
            'tag:yaml.org,2002:python/module:' + data.__name__,
            '',
        )

    def represent_object(self, data):
        data_type = type(data)

        if data_type in copyreg.dispatch_table:
            reduction = copyreg.dispatch_table[data_type](data)
        elif hasattr(data, '__reduce_ex__'):
            reduction = data.__reduce_ex__(2)
        elif hasattr(data, '__reduce__'):
            reduction = data.__reduce__()
        else:
            raise RepresenterError("cannot represent an object", data)

        if isinstance(reduction, str):
            return self.represent_scalar(
                'tag:yaml.org,2002:python/name:' + reduction,
                '',
            )

        reduction = (list(reduction) + [None] * 5)[:5]
        function, args, state, listitems, dictitems = reduction
        args = list(args)

        if state is None:
            state = {}

        if listitems is not None:
            listitems = list(listitems)

        if dictitems is not None:
            dictitems = list(dictitems)

        if function.__name__ == '__newobj__':
            function = args[0]
            args = args[1:]
            tag = 'tag:yaml.org,2002:python/object/new:'
            is_new = True
        else:
            tag = 'tag:yaml.org,2002:python/object/apply:'
            is_new = False

        function_name = '%s.%s' % (function.__module__, function.__name__)

        if (not args and not listitems and not dictitems
                and isinstance(state, dict) and not is_new):
            return self.represent_mapping(
                'tag:yaml.org,2002:python/object:' + function_name,
                state,
            )

        if (not listitems and not dictitems
                and isinstance(state, dict) and is_new):
            return self.represent_mapping(
                'tag:yaml.org,2002:python/object/new:' + function_name,
                state,
            )

        value = {}

        if args:
            value['args'] = args

        if state or not isinstance(state, dict):
            value['state'] = state

        if listitems:
            value['listitems'] = listitems

        if dictitems:
            value['dictitems'] = dictitems

        return self.represent_mapping(tag + function_name, value)

    def represent_ordered_dict(self, data):
        value = []
        for key, item in data.items():
            value.append(self.represent_mapping(
                'tag:yaml.org,2002:map',
                [(key, item)],
            ))
        return SequenceNode('tag:yaml.org,2002:omap', value)


Representer.add_representer(complex, Representer.represent_complex)
Representer.add_representer(tuple, Representer.represent_tuple)
Representer.add_representer(type, Representer.represent_name)
Representer.add_representer(types.FunctionType, Representer.represent_name)
Representer.add_representer(
    types.BuiltinFunctionType,
    Representer.represent_name,
)
Representer.add_representer(types.ModuleType, Representer.represent_module)
Representer.add_representer(
    collections.OrderedDict,
    Representer.represent_ordered_dict,
)
Representer.add_multi_representer(object, Representer.represent_object)
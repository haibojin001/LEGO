from .error import *
from .nodes import *

import base64
import binascii
import collections.abc
import datetime
import re
import sys
import types


__all__ = [
    'BaseConstructor',
    'SafeConstructor',
    'FullConstructor',
    'UnsafeConstructor',
    'Constructor',
    'ConstructorError',
]


class ConstructorError(MarkedYAMLError):
    pass


class BaseConstructor:

    yaml_constructors = {}
    yaml_multi_constructors = {}

    def __init__(self):
        self.constructed_objects = {}
        self.recursive_objects = {}
        self.state_generators = []
        self.deep_construct = False

    def check_data(self):
        return self.check_node()

    def check_state_key(self, key):
        if self.get_state_keys_blacklist_regexp().match(key):
            raise ConstructorError(
                None, None,
                "blacklisted key '%s' in instance state found" % key,
                None
            )

    def get_data(self):
        if self.check_node():
            return self.construct_document(self.get_node())

    def get_single_data(self):
        node = self.get_single_node()
        if node is not None:
            return self.construct_document(node)
        return None

    def construct_document(self, node):
        data = self.construct_object(node)
        while self.state_generators:
            generators = self.state_generators
            self.state_generators = []
            for generator in generators:
                for _ in generator:
                    pass
        self.constructed_objects = {}
        self.recursive_objects = {}
        self.deep_construct = False
        return data

    def construct_object(self, node, deep=False):
        if node in self.constructed_objects:
            return self.constructed_objects[node]

        old_deep = None
        if deep:
            old_deep = self.deep_construct
            self.deep_construct = True

        if node in self.recursive_objects:
            raise ConstructorError(
                None, None,
                "found unconstructable recursive node",
                node.start_mark
            )

        self.recursive_objects[node] = None
        constructor = None
        tag_suffix = None

        if node.tag in self.yaml_constructors:
            constructor = self.yaml_constructors[node.tag]
        else:
            for prefix, multi_constructor in self.yaml_multi_constructors.items():
                if prefix is not None and node.tag.startswith(prefix):
                    tag_suffix = node.tag[len(prefix):]
                    constructor = multi_constructor
                    break
            else:
                if None in self.yaml_multi_constructors:
                    tag_suffix = node.tag
                    constructor = self.yaml_multi_constructors[None]
                elif None in self.yaml_constructors:
                    constructor = self.yaml_constructors[None]
                elif isinstance(node, ScalarNode):
                    constructor = self.__class__.construct_scalar
                elif isinstance(node, SequenceNode):
                    constructor = self.__class__.construct_sequence
                elif isinstance(node, MappingNode):
                    constructor = self.__class__.construct_mapping

        if tag_suffix is None:
            data = constructor(self, node)
        else:
            data = constructor(self, tag_suffix, node)

        if isinstance(data, types.GeneratorType):
            generator = data
            data = next(generator)
            if self.deep_construct:
                for _ in generator:
                    pass
            else:
                self.state_generators.append(generator)

        self.constructed_objects[node] = data
        del self.recursive_objects[node]

        if deep:
            self.deep_construct = old_deep

        return data

    def construct_scalar(self, node):
        if not isinstance(node, ScalarNode):
            raise ConstructorError(
                None, None,
                "expected a scalar node, but found %s" % node.id,
                node.start_mark
            )
        return node.value

    def construct_sequence(self, node, deep=False):
        if not isinstance(node, SequenceNode):
            raise ConstructorError(
                None, None,
                "expected a sequence node, but found %s" % node.id,
                node.start_mark
            )
        return [self.construct_object(child, deep=deep) for child in node.value]

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None, None,
                "expected a mapping node, but found %s" % node.id,
                node.start_mark
            )

        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, collections.abc.Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping

    def construct_pairs(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None, None,
                "expected a mapping node, but found %s" % node.id,
                node.start_mark
            )

        pairs = []
        for key_node, value_node in node.value:
            pairs.append((
                self.construct_object(key_node, deep=deep),
                self.construct_object(value_node, deep=deep),
            ))
        return pairs

    @classmethod
    def add_constructor(cls, tag, constructor):
        if 'yaml_constructors' not in cls.__dict__:
            cls.yaml_constructors = cls.yaml_constructors.copy()
        cls.yaml_constructors[tag] = constructor

    @classmethod
    def add_multi_constructor(cls, tag_prefix, multi_constructor):
        if 'yaml_multi_constructors' not in cls.__dict__:
            cls.yaml_multi_constructors = cls.yaml_multi_constructors.copy()
        cls.yaml_multi_constructors[tag_prefix] = multi_constructor


class SafeConstructor(BaseConstructor):

    bool_values = {
        'yes': True, 'no': False,
        'true': True, 'false': False,
        'on': True, 'off': False,
    }

    inf_value = 1e300 * 1e300
    nan_value = inf_value / inf_value

    timestamp_regexp = re.compile(r"""
        ^(?P<year>[0-9][0-9][0-9][0-9])
        -(?P<month>[0-9][0-9]?)
        -(?P<day>[0-9][0-9]?)
        (?:
            [Tt]|[ \t]+
            (?P<hour>[0-9][0-9]?)
            :(?P<minute>[0-9][0-9])
            :(?P<second>[0-9][0-9])
            (?:\.(?P<fraction>[0-9]*))?
            (?:
                [ \t]*
                (?P<tz>Z|(?P<tz_sign>[-+])
                (?P<tz_hour>[0-9][0-9]?)
                (?::(?P<tz_minute>[0-9][0-9])?)?)
            )?
        )?$
    """, re.X)

    def construct_scalar(self, node):
        if isinstance(node, MappingNode):
            for key_node, value_node in node.value:
                if key_node.tag == 'tag:yaml.org,2002:value':
                    return self.construct_scalar(value_node)
        return super().construct_scalar(node)

    def flatten_mapping(self, node):
        merge = []
        merge_key_ids = set()
        index = 0

        while index < len(node.value):
            key_node, value_node = node.value[index]

            if key_node.tag == 'tag:yaml.org,2002:merge':
                del node.value[index]

                if isinstance(value_node, MappingNode):
                    self.flatten_mapping(value_node)
                    for pair in value_node.value:
                        key_id = id(pair[0].value)
                        if key_id not in merge_key_ids:
                            merge_key_ids.add(key_id)
                            merge.append(pair)

                elif isinstance(value_node, SequenceNode):
                    for subnode in value_node.value:
                        if not isinstance(subnode, MappingNode):
                            raise ConstructorError(
                                "while constructing a mapping",
                                node.start_mark,
                                "expected a mapping for merging, but found %s"
                                % subnode.id,
                                subnode.start_mark
                            )

                        self.flatten_mapping(subnode)
                        for pair in subnode.value:
                            key_id = id(pair[0].value)
                            if key_id not in merge_key_ids:
                                merge_key_ids.add(key_id)
                                merge.append(pair)

                else:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "expected a mapping or list of mappings for merging, but found %s"
                        % value_node.id,
                        value_node.start_mark
                    )

            elif key_node.tag == 'tag:yaml.org,2002:value':
                key_node.tag = 'tag:yaml.org,2002:str'
                index += 1
            else:
                index += 1

        if merge:
            node.value = merge + node.value

    def construct_mapping(self, node, deep=False):
        if isinstance(node, MappingNode):
            self.flatten_mapping(node)
        return super().construct_mapping(node, deep=deep)

    def construct_yaml_null(self, node):
        self.construct_scalar(node)
        return None

    def construct_yaml_bool(self, node):
        value = self.construct_scalar(node)
        return self.bool_values[value.lower()]

    def construct_yaml_int(self, node):
        value = self.construct_scalar(node)
        value = value.replace('_', '')
        sign = +1

        if value[0] == '-':
            sign = -1
        if value[0] in '+-':
            value = value[1:]

        if value == '0':
            return 0

        if value.startswith('0b'):
            return sign * int(value[2:], 2)

        if value.startswith('0x'):
            return sign * int(value[2:], 16)

        if value.startswith('0o'):
            return sign * int(value[2:], 8)

        if value[0] == '0':
            return sign * int(value, 8)

        if ':' in value:
            digits = [int(part) for part in value.split(':')]
            digits.reverse()
            number = 0
            base = 1
            for digit in digits:
                number += digit * base
                base *= 60
            return sign * number

        return sign * int(value)

    def construct_yaml_float(self, node):
        value = self.construct_scalar(node)
        value = value.replace('_', '').lower()
        sign = +1

        if value[0] == '-':
            sign = -1
        if value[0] in '+-':
            value = value[1:]

        if value == '.inf':
            return sign * self.inf_value

        if value == '.nan':
            return self.nan_value

        if ':' in value:
            digits = [float(part) for part in value.split(':')]
            digits.reverse()
            number = 0.0
            base = 1
            for digit in digits:
                number += digit * base
                base *= 60
            return sign * number

        return sign * float(value)

    def construct_yaml_binary(self, node):
        value = self.construct_scalar(node)
        try:
            return base64.decodebytes(value.encode('ascii'))
        except UnicodeEncodeError as exc:
            raise ConstructorError(
                None, None,
                "failed to convert base64 data into ASCII: %s" % exc,
                node.start_mark
            )
        except binascii.Error as exc:
            raise ConstructorError(
                None, None,
                "failed to decode base64 data: %s" % exc,
                node.start_mark
            )

    def construct_yaml_timestamp(self, node):
        value = self.construct_scalar(node)
        match = self.timestamp_regexp.match(value)

        values = match.groupdict()
        year = int(values['year'])
        month = int(values['month'])
        day = int(values['day'])

        if not values['hour']:
            return datetime.date(year, month, day)

        hour = int(values['hour'])
        minute = int(values['minute'])
        second = int(values['second'])
        fraction = values['fraction']

        if fraction:
            fraction = int((fraction + '000000')[:6])
        else:
            fraction = 0

        tz = values['tz']
        if not tz:
            tzinfo = None
        elif tz == 'Z':
            tzinfo = datetime.timezone.utc
        else:
            tz_hour = int(values['tz_hour'])
            tz_minute = int(values['tz_minute'] or 0)
            delta = datetime.timedelta(hours=tz_hour, minutes=tz_minute)
            if values['tz_sign'] == '-':
                delta = -delta
            tzinfo = datetime.timezone(delta)

        return datetime.datetime(
            year, month, day, hour, minute, second, fraction, tzinfo
        )

    def construct_yaml_omap(self, node):
        omap = []
        yield omap

        if not isinstance(node, SequenceNode):
            raise ConstructorError(
                "while constructing an ordered map",
                node.start_mark,
                "expected a sequence, but found %s" % node.id,
                node.start_mark
            )

        for subnode in node.value:
            if not isinstance(subnode, MappingNode):
                raise ConstructorError(
                    "while constructing an ordered map",
                    node.start_mark,
                    "expected a mapping of length 1, but found %s" % subnode.id,
                    subnode.start_mark
                )

            if len(subnode.value) != 1:
                raise ConstructorError(
                    "while constructing an ordered map",
                    node.start_mark,
                    "expected a single mapping item, but found %d items"
                    % len(subnode.value),
                    subnode.start_mark
                )

            key_node, value_node = subnode.value[0]
            key = self.construct_object(key_node)
            value = self.construct_object(value_node)
            omap.append((key, value))

    def construct_yaml_pairs(self, node):
        pairs = []
        yield pairs

        if not isinstance(node, SequenceNode):
            raise ConstructorError(
                "while constructing pairs",
                node.start_mark,
                "expected a sequence, but found %s" % node.id,
                node.start_mark
            )

        for subnode in node.value:
            if not isinstance(subnode, MappingNode):
                raise ConstructorError(
                    "while constructing pairs",
                    node.start_mark,
                    "expected a mapping of length 1, but found %s" % subnode.id,
                    subnode.start_mark
                )

            if len(subnode.value) != 1:
                raise ConstructorError(
                    "while constructing pairs",
                    node.start_mark,
                    "expected a single mapping item, but found %d items"
                    % len(subnode.value),
                    subnode.start_mark
                )

            key_node, value_node = subnode.value[0]
            pairs.append((
                self.construct_object(key_node),
                self.construct_object(value_node),
            ))

    def construct_yaml_set(self, node):
        data = set()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_yaml_str(self, node):
        return self.construct_scalar(node)

    def construct_yaml_seq(self, node):
        data = []
        yield data
        data.extend(self.construct_sequence(node))

    def construct_yaml_map(self, node):
        data = {}
        yield data
        data.update(self.construct_mapping(node))

    def construct_yaml_object(self, node):
        data = {}
        yield data
        data.update(self.construct_mapping(node))


SafeConstructor.add_constructor(
    'tag:yaml.org,2002:null',
    SafeConstructor.construct_yaml_null,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:bool',
    SafeConstructor.construct_yaml_bool,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:int',
    SafeConstructor.construct_yaml_int,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:float',
    SafeConstructor.construct_yaml_float,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:binary',
    SafeConstructor.construct_yaml_binary,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:timestamp',
    SafeConstructor.construct_yaml_timestamp,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:omap',
    SafeConstructor.construct_yaml_omap,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:pairs',
    SafeConstructor.construct_yaml_pairs,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:set',
    SafeConstructor.construct_yaml_set,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:str',
    SafeConstructor.construct_yaml_str,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:seq',
    SafeConstructor.construct_yaml_seq,
)
SafeConstructor.add_constructor(
    'tag:yaml.org,2002:map',
    SafeConstructor.construct_yaml_map,
)
SafeConstructor.add_constructor(
    None,
    SafeConstructor.construct_yaml_str,
)


class FullConstructor(SafeConstructor):

    def construct_python_str(self, node):
        return self.construct_scalar(node)

    def construct_python_unicode(self, node):
        return self.construct_scalar(node)

    def construct_python_bytes(self, node):
        return self.construct_yaml_binary(node)

    def construct_python_long(self, node):
        return self.construct_yaml_int(node)

    def construct_python_complex(self, node):
        return complex(self.construct_scalar(node))

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

    def find_python_module(self, name, mark, unsafe=False):
        if not name:
            raise ConstructorError(
                "while constructing a Python module",
                mark,
                "expected non-empty name",
                mark
            )

        if unsafe:
            __import__(name)

        if name not in sys.modules:
            raise ConstructorError(
                "while constructing a Python module",
                mark,
                "module %r is not imported" % name,
                mark
            )

        return sys.modules[name]

    def find_python_name(self, name, mark, unsafe=False):
        if not name:
            raise ConstructorError(
                "while constructing a Python object",
                mark,
                "expected non-empty name",
                mark
            )

        if '.' in name:
            module_name, object_name = name.rsplit('.', 1)
        else:
            module_name = 'builtins'
            object_name = name

        module = self.find_python_module(module_name, mark, unsafe=unsafe)

        if not hasattr(module, object_name):
            raise ConstructorError(
                "while constructing a Python object",
                mark,
                "cannot find %r in the module %r" % (object_name, module_name),
                mark
            )

        return getattr(module, object_name)

    def construct_python_name(self, suffix, node):
        value = self.construct_scalar(node)
        if value:
            raise ConstructorError(
                "while constructing a Python name",
                node.start_mark,
                "expected the empty value, but found %r" % value,
                node.start_mark
            )
        return self.find_python_name(suffix, node.start_mark)

    def construct_python_module(self, suffix, node):
        value = self.construct_scalar(node)
        if value:
            raise ConstructorError(
                "while constructing a Python module",
                node.start_mark,
                "expected the empty value, but found %r" % value,
                node.start_mark
            )
        return self.find_python_module(suffix, node.start_mark)

    def make_python_instance(
        self, suffix, node, args=None, kwds=None, newobj=False, unsafe=False
    ):
        if args is None:
            args = []
        if kwds is None:
            kwds = {}

        cls = self.find_python_name(suffix, node.start_mark, unsafe=unsafe)

        if not (unsafe or isinstance(cls, type)):
            raise ConstructorError(
                "while constructing a Python instance",
                node.start_mark,
                "expected a class, but found %r" % type(cls),
                node.start_mark
            )

        if newobj:
            return cls.__new__(cls, *args, **kwds)
        return cls(*args, **kwds)

    def set_python_instance_state(self, instance, state, unsafe=False):
        if hasattr(instance, '__setstate__'):
            instance.__setstate__(state)
            return

        slotstate = {}
        if isinstance(state, tuple):
            state, slotstate = state

        if state:
            if not unsafe:
                for key in state:
                    self.check_state_key(key)
            instance.__dict__.update(state)

        if slotstate:
            for key, value in slotstate.items():
                if not unsafe:
                    self.check_state_key(key)
                setattr(instance, key, value)

    def construct_python_object(self, suffix, node):
        instance = self.make_python_instance(suffix, node, newobj=True)
        yield instance
        state = self.construct_mapping(node, deep=True)
        self.set_python_instance_state(instance, state)

    def construct_python_object_apply(self, suffix, node, newobj=False):
        if isinstance(node, SequenceNode):
            args = self.construct_sequence(node, deep=True)
            kwds = {}
            state = {}
            listitems = []
            dictitems = []
        else:
            value = self.construct_mapping(node, deep=True)
            args = value.pop('args', [])
            kwds = value.pop('kwds', {})
            state = value.pop('state', {})
            listitems = value.pop('listitems', [])
            dictitems = value.pop('dictitems', [])

        instance = self.make_python_instance(
            suffix, node, args, kwds, newobj=newobj
        )
        yield instance

        if state:
            self.set_python_instance_state(instance, state)

        if listitems:
            instance.extend(listitems)

        if dictitems:
            for key, value in dictitems:
                instance[key] = value

    def construct_python_object_new(self, suffix, node):
        return self.construct_python_object_apply(suffix, node, newobj=True)

    def get_state_keys_blacklist(self):
        return ['^extend$', '^__.*__$']

    def get_state_keys_blacklist_regexp(self):
        if not hasattr(self, 'state_keys_blacklist_regexp'):
            self.state_keys_blacklist_regexp = re.compile(
                '(' + '|'.join(self.get_state_keys_blacklist()) + ')'
            )
        return self.state_keys_blacklist_regexp


FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/none',
    FullConstructor.construct_yaml_null,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/bool',
    FullConstructor.construct_yaml_bool,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/str',
    FullConstructor.construct_python_str,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/unicode',
    FullConstructor.construct_python_unicode,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/bytes',
    FullConstructor.construct_python_bytes,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/int',
    FullConstructor.construct_yaml_int,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/long',
    FullConstructor.construct_python_long,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/float',
    FullConstructor.construct_yaml_float,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/complex',
    FullConstructor.construct_python_complex,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/list',
    FullConstructor.construct_yaml_seq,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    FullConstructor.construct_python_tuple,
)
FullConstructor.add_constructor(
    'tag:yaml.org,2002:python/dict',
    FullConstructor.construct_yaml_map,
)
FullConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/name:',
    FullConstructor.construct_python_name,
)
FullConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/module:',
    FullConstructor.construct_python_module,
)
FullConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/object:',
    FullConstructor.construct_python_object,
)
FullConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/object/new:',
    FullConstructor.construct_python_object_new,
)
FullConstructor.add_multi_constructor(
    'tag:yaml.org,2002:python/object/apply:',
    FullConstructor.construct_python_object_apply,
)


class UnsafeConstructor(FullConstructor):

    def find_python_module(self, name, mark):
        return super().find_python_module(name, mark, unsafe=True)

    def find_python_name(self, name, mark):
        return super().find_python_name(name, mark, unsafe=True)

    def make_python_instance(
        self, suffix, node, args=None, kwds=None, newobj=False
    ):
        return super().make_python_instance(
            suffix, node, args, kwds, newobj, unsafe=True
        )

    def set_python_instance_state(self, instance, state):
        return super().set_python_instance_state(instance, state, unsafe=True)


Constructor = UnsafeConstructor
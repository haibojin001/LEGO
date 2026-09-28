from html import escape as html_escape
import types
from itertools import islice
from collections.abc import Sequence, Mapping, MutableSequence

try:
    from .typeutils import make_sentinel
    _MISSING = make_sentinel(var_name='_MISSING')
except ImportError:
    _MISSING = object()


__all__ = ['Table']


def to_text(obj, maxlen=None):
    try:
        value = str(obj)
    except Exception:
        try:
            value = str(repr(obj))
        except Exception:
            value = str(object.__repr__(obj))
    if maxlen and len(value) > maxlen:
        value = value[:maxlen - 3] + '...'
    return value


def escape_html(obj, maxlen=None):
    return html_escape(to_text(obj, maxlen=maxlen), quote=True)


_DNR = {
    type(None), bool, complex, float, type(NotImplemented), slice,
    str, bytes, int,
    types.FunctionType, types.MethodType,
    types.BuiltinFunctionType, types.GeneratorType,
}


class UnsupportedData(TypeError):
    pass


class InputType:
    def __init__(self, *args, **kwargs):
        pass

    def get_entry_seq(self, data_seq, headers):
        return [self.get_entry(entry, headers) for entry in data_seq]


class DictInputType(InputType):
    def check_type(self, obj):
        return isinstance(obj, Mapping)

    def guess_headers(self, obj):
        return sorted(obj.keys())

    def get_entry(self, obj, headers):
        return [obj.get(header) for header in headers]

    def get_entry_seq(self, data_seq, headers):
        return [[entry.get(header) for header in headers] for entry in data_seq]


class ObjectInputType(InputType):
    def check_type(self, obj):
        return type(obj) not in _DNR and hasattr(obj, '__class__')

    def guess_headers(self, obj):
        ret = []
        for attr in dir(obj):
            try:
                value = getattr(obj, attr)
            except Exception:
                continue
            if not callable(value):
                ret.append(attr)
        return ret

    def get_entry(self, obj, headers):
        ret = []
        for header in headers:
            try:
                ret.append(getattr(obj, header))
            except Exception:
                ret.append(None)
        return ret


class ListInputType(InputType):
    def check_type(self, obj):
        return isinstance(obj, MutableSequence)

    def guess_headers(self, obj):
        return None

    def get_entry(self, obj, headers):
        return obj

    def get_entry_seq(self, data_seq, headers):
        return data_seq


class TupleInputType(InputType):
    def check_type(self, obj):
        return isinstance(obj, tuple)

    def guess_headers(self, obj):
        return None

    def get_entry(self, obj, headers):
        return list(obj)

    def get_entry_seq(self, data_seq, headers):
        return [list(entry) for entry in data_seq]


class NamedTupleInputType(InputType):
    def check_type(self, obj):
        return isinstance(obj, tuple) and hasattr(obj, '_fields')

    def guess_headers(self, obj):
        return list(obj._fields)

    def get_entry(self, obj, headers):
        return [getattr(obj, header, None) for header in headers]

    def get_entry_seq(self, data_seq, headers):
        return [[getattr(entry, header, None) for header in headers]
                for entry in data_seq]


class Table:
    _input_types = [
        DictInputType(),
        ListInputType(),
        NamedTupleInputType(),
        TupleInputType(),
        ObjectInputType(),
    ]

    _html_tr, _html_tr_close = '<tr>', '</tr>'
    _html_th, _html_th_close = '<th>', '</th>'
    _html_td, _html_td_close = '<td>', '</td>'
    _html_thead, _html_thead_close = '<thead>', '</thead>'
    _html_tbody, _html_tbody_close = '<tbody>', '</tbody>'
    _html_table_tag, _html_table_tag_close = '<table>', '</table>'

    def __init__(self, data=None, headers=_MISSING, metadata=None):
        if headers is _MISSING:
            headers = []
            if data:
                headers, data = list(data[0]), islice(data, 1, None)
        self.headers = headers or []
        self.metadata = metadata or {}
        self._data = []
        if data:
            self.extend(data)

    @property
    def data(self):
        return self._data

    @property
    def width(self):
        widths = [len(self.headers)]
        if self._data:
            widths.extend(len(row) for row in self._data)
        return max(widths) if widths else 0

    @property
    def height(self):
        return len(self._data)

    def __repr__(self):
        return '%s(headers=%r, data=%r)' % (
            self.__class__.__name__, self.headers, self._data)

    @classmethod
    def from_data(cls, data, headers=_MISSING, max_depth=1, **kwargs):
        if data is None:
            return cls(headers=headers, **kwargs)

        if isinstance(data, Mapping) or not isinstance(data, Sequence):
            if isinstance(data, (str, bytes)):
                raise UnsupportedData('unsupported data type: %r' % type(data))
            try:
                if not isinstance(data, Mapping):
                    data = list(data)
            except TypeError:
                data = [data]
            else:
                if not data:
                    return cls(headers=headers, **kwargs)
                if not isinstance(data[0], (Mapping, MutableSequence, tuple)):
                    data = [data]
        elif not data:
            return cls(headers=headers, **kwargs)

        if isinstance(data, Mapping):
            data = [data]
        elif not isinstance(data, Sequence):
            data = list(data)

        if not data:
            return cls(headers=headers, **kwargs)

        sample = data[0]
        input_type = None
        for candidate in cls._input_types:
            if candidate.check_type(sample):
                input_type = candidate
                break
        if input_type is None:
            raise UnsupportedData('unsupported data type: %r' % type(sample))

        if headers is _MISSING:
            headers = input_type.guess_headers(sample)

        rows = input_type.get_entry_seq(data, headers)

        if max_depth is not None and max_depth > 1:
            nested_depth = max_depth - 1
            converted = []
            for row in rows:
                new_row = []
                for value in row:
                    if isinstance(value, Table):
                        new_row.append(value)
                        continue
                    if isinstance(value, Mapping):
                        try:
                            new_row.append(cls.from_data(
                                value, max_depth=nested_depth))
                            continue
                        except UnsupportedData:
                            pass
                    if (isinstance(value, Sequence)
                            and not isinstance(value, (str, bytes))):
                        try:
                            new_row.append(cls.from_data(
                                value, max_depth=nested_depth))
                            continue
                        except (UnsupportedData, TypeError):
                            pass
                    new_row.append(value)
                converted.append(new_row)
            rows = converted

        return cls(rows, headers=headers, **kwargs)

    def _fill(self):
        width = self.width
        if len(self.headers) < width:
            self.headers.extend([None] * (width - len(self.headers)))
        for row in self._data:
            if len(row) < width:
                row.extend([None] * (width - len(row)))

    def extend(self, data):
        if data is None:
            return
        if not isinstance(data, Sequence):
            data = list(data)
        if not data:
            return

        for row in data:
            if not isinstance(row, MutableSequence):
                raise TypeError('expected a sequence of mutable row sequences')
        self._data.extend(data)
        self._fill()

    def get_cell_html(self, value, max_depth=1):
        if isinstance(value, Table) and max_depth:
            return value.to_html(max_depth=max_depth - 1)
        return escape_html(value)

    def get_cell_text(self, value, maxlen=None):
        if isinstance(value, Table):
            return value.to_text(max_cell_width=maxlen)
        return to_text(value, maxlen=maxlen)

    def to_html(self, orientation='auto', wrapped=True, with_headers=True,
                with_newlines=True, max_depth=1):
        if orientation not in ('auto', 'horizontal', 'vertical'):
            raise ValueError('expected orientation to be "auto", "horizontal",'
                             ' or "vertical"')

        width = self.width
        if orientation == 'auto':
            orientation = 'horizontal'

        newline = '\n' if with_newlines else ''
        parts = []

        if wrapped:
            parts.append(self._html_table_tag)

        if orientation == 'horizontal':
            if with_headers and self.headers:
                parts.append(self._html_thead)
                cells = ''.join(
                    self._html_th + self.get_cell_html(header, max_depth)
                    + self._html_th_close
                    for header in self.headers)
                parts.append(self._html_tr + cells + self._html_tr_close)
                parts.append(self._html_thead_close)

            parts.append(self._html_tbody)
            for row in self._data:
                cells = ''.join(
                    self._html_td + self.get_cell_html(value, max_depth)
                    + self._html_td_close
                    for value in row[:width])
                parts.append(self._html_tr + cells + self._html_tr_close)
            parts.append(self._html_tbody_close)
        else:
            parts.append(self._html_tbody)
            for col_idx in range(width):
                cells = []
                if with_headers and self.headers:
                    header = self.headers[col_idx] if col_idx < len(self.headers) else None
                    cells.append(self._html_th + self.get_cell_html(header, max_depth)
                                 + self._html_th_close)
                for row in self._data:
                    value = row[col_idx] if col_idx < len(row) else None
                    cells.append(self._html_td + self.get_cell_html(value, max_depth)
                                 + self._html_td_close)
                parts.append(self._html_tr + ''.join(cells) + self._html_tr_close)
            parts.append(self._html_tbody_close)

        if wrapped:
            parts.append(self._html_table_tag_close)
        return newline.join(parts)

    render_html = to_html

    def to_text(self, headers=True, max_width=None, max_cell_width=None):
        width = self.width
        if not width:
            return ''

        if max_cell_width is None and max_width is not None:
            max_cell_width = max(1, max_width // width - 3)

        rows = []
        if headers and self.headers:
            rows.append([self.get_cell_text(
                self.headers[idx] if idx < len(self.headers) else None,
                max_cell_width)
                for idx in range(width)])

        body_rows = [
            [self.get_cell_text(
                row[idx] if idx < len(row) else None, max_cell_width)
             for idx in range(width)]
            for row in self._data
        ]

        all_rows = rows + body_rows
        if not all_rows:
            return ''

        col_widths = [0] * width
        for row in all_rows:
            for index, cell in enumerate(row):
                line_width = max((len(line) for line in cell.splitlines()),
                                 default=0)
                if line_width > col_widths[index]:
                    col_widths[index] = line_width

        def format_row(row):
            line_sets = [cell.splitlines() or [''] for cell in row]
            line_count = max(len(lines) for lines in line_sets)
            result = []
            for line_idx in range(line_count):
                result.append(' | '.join(
                    (lines[line_idx] if line_idx < len(lines) else '').ljust(
                        col_widths[col_idx])
                    for col_idx, lines in enumerate(line_sets)).rstrip())
            return result

        ret = []
        if rows:
            ret.extend(format_row(rows[0]))
            ret.append('-+-'.join('-' * cell_width for cell_width in col_widths))
        for row in body_rows:
            ret.extend(format_row(row))
        return '\n'.join(ret)

    render_text = to_text
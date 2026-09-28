import os
import math
import time
import codecs
import random
import itertools
import base64
from collections import deque
from collections.abc import Mapping, Sequence, Set, ItemsView, Iterable
from itertools import zip_longest

try:
    from .typeutils import make_sentinel
    _UNSET = make_sentinel('_UNSET')
    _REMAP_EXIT = make_sentinel('_REMAP_EXIT')
except Exception:
    _UNSET = object()
    _REMAP_EXIT = object()


def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def is_scalar(obj):
    return not is_iterable(obj) or isinstance(obj, (str, bytes))


def is_collection(obj):
    return is_iterable(obj) and not isinstance(obj, (str, bytes))


def split(src, sep=None, maxsplit=None):
    return list(split_iter(src, sep, maxsplit))


def split_iter(src, sep=None, maxsplit=None):
    if not is_iterable(src):
        raise TypeError('expected an iterable')

    if maxsplit is not None:
        maxsplit = int(maxsplit)
        if maxsplit == 0:
            yield list(src)
            return

    if callable(sep):
        is_sep = sep
    elif not is_scalar(sep):
        separators = frozenset(sep)

        def is_sep(value):
            return value in separators
    else:
        def is_sep(value):
            return value == sep

    current = []
    split_count = 0
    for value in src:
        if maxsplit is not None and split_count >= maxsplit:
            current.append(value)
            continue
        if is_sep(value):
            if sep is None and not current:
                continue
            split_count += 1
            yield current
            current = []
        else:
            current.append(value)

    if current or sep is not None:
        yield current


def lstrip(iterable, strip_value=None):
    return list(lstrip_iter(iterable, strip_value))


def lstrip_iter(iterable, strip_value=None):
    iterator = iter(iterable)
    for value in iterator:
        if value != strip_value:
            yield value
            break
    yield from iterator


def rstrip(iterable, strip_value=None):
    return list(rstrip_iter(iterable, strip_value))


def rstrip_iter(iterable, strip_value=None):
    trailing = []
    for value in iterable:
        if value == strip_value:
            trailing.append(value)
        else:
            if trailing:
                yield from trailing
                trailing = []
            yield value


def strip(iterable, strip_value=None):
    return list(strip_iter(iterable, strip_value))


def strip_iter(iterable, strip_value=None):
    return rstrip_iter(lstrip_iter(iterable, strip_value), strip_value)


def chunked(src, size, **kw):
    return list(chunked_iter(src, size, **kw))


def chunked_iter(src, size, **kw):
    if not is_iterable(src):
        raise TypeError('expected an iterable')
    if not isinstance(size, int):
        raise TypeError('expected an integer size')
    if size <= 0:
        raise ValueError('expected a positive size')

    fill = kw.pop('fill', _UNSET)
    if kw:
        name = next(iter(kw))
        raise TypeError("unexpected keyword argument %r" % name)

    iterator = iter(src)
    while True:
        group = list(itertools.islice(iterator, size))
        if not group:
            return
        if len(group) < size and fill is not _UNSET:
            group.extend([fill] * (size - len(group)))
        yield group


def chunk_ranges(input_size, chunk_size, input_offset=0, overlap_size=0, align=False):
    if input_size < 0:
        raise ValueError('expected a non-negative input size')
    if chunk_size <= 0:
        raise ValueError('expected a positive chunk size')
    if overlap_size < 0:
        raise ValueError('expected a non-negative overlap size')
    if overlap_size >= chunk_size:
        raise ValueError('overlap size must be less than chunk size')

    input_size = int(input_size)
    chunk_size = int(chunk_size)
    input_offset = int(input_offset)
    overlap_size = int(overlap_size)

    stop = input_offset + input_size
    if align:
        start = input_offset - (input_offset % chunk_size)
        if start < input_offset:
            start = max(input_offset, start)
    else:
        start = input_offset

    while start < stop:
        end = min(start + chunk_size, stop)
        yield start, end
        if end == stop:
            break
        start = end - overlap_size


def chunked_even(src, count):
    return list(chunked_even_iter(src, count))


def chunked_even_iter(src, count):
    if not isinstance(count, int):
        raise TypeError('expected an integer count')
    if count <= 0:
        raise ValueError('expected a positive count')

    values = list(src)
    quotient, remainder = divmod(len(values), count)
    index = 0
    for group_num in range(count):
        width = quotient + (group_num < remainder)
        if width:
            yield values[index:index + width]
        index += width


def pairwise(src, end=_UNSET):
    return windowed(src, 2, fill=end)


def pairwise_iter(src, end=_UNSET):
    return windowed_iter(src, 2, fill=end)


def windowed(src, size, fill=_UNSET):
    return list(windowed_iter(src, size, fill))


def windowed_iter(src, size, fill=_UNSET):
    if not isinstance(size, int):
        raise TypeError('expected integer size')
    if size <= 0:
        raise ValueError('expected a positive size')

    iterator = iter(src)
    window = deque(itertools.islice(iterator, size), maxlen=size)
    if not window:
        return
    if len(window) < size:
        if fill is _UNSET:
            return
        window.extend([fill] * (size - len(window)))
        yield tuple(window)
        return

    yield tuple(window)
    for value in iterator:
        window.append(value)
        yield tuple(window)

    if fill is not _UNSET:
        for _ in range(size - 1):
            window.append(fill)
            yield tuple(window)


def partition(src, key=bool):
    truthy = []
    falsy = []
    for value in src:
        if key(value):
            truthy.append(value)
        else:
            falsy.append(value)
    return truthy, falsy


def partition_iter(src, key=bool):
    iterator_a, iterator_b = itertools.tee(src)
    return (
        (value for value in iterator_a if key(value)),
        (value for value in iterator_b if not key(value)),
    )


def unique(src, key=None):
    return list(unique_iter(src, key))


def unique_iter(src, key=None):
    seen = set()
    if key is None:
        for value in src:
            if value not in seen:
                seen.add(value)
                yield value
    else:
        for value in src:
            marker = key(value)
            if marker not in seen:
                seen.add(marker)
                yield value


def redundant(src, key=None, groups=False):
    return list(redundant_iter(src, key, groups))


def redundant_iter(src, key=None, groups=False):
    if groups:
        grouped = {}
        order = []
        for value in src:
            marker = value if key is None else key(value)
            if marker not in grouped:
                grouped[marker] = [value]
                order.append(marker)
            else:
                grouped[marker].append(value)
        for marker in order:
            group = grouped[marker]
            if len(group) > 1:
                yield group
        return

    seen = set()
    for value in src:
        marker = value if key is None else key(value)
        if marker in seen:
            yield value
        else:
            seen.add(marker)


def one(src, default=None, key=None):
    if key is None:
        iterator = iter(src)
        for value in iterator:
            return value
        return default

    for value in src:
        if key(value):
            return value
    return default


def first(src, default=None, key=None):
    return one(src, default, key)


def flatten(iterable):
    return list(flatten_iter(iterable))


def flatten_iter(iterable):
    for value in iterable:
        if is_scalar(value):
            yield value
        else:
            yield from flatten_iter(value)


def same(iterable, ref=_UNSET):
    iterator = iter(iterable)
    if ref is _UNSET:
        try:
            ref = next(iterator)
        except StopIteration:
            return True
    return all(value == ref for value in iterator)


def default_visit(path, key, value):
    return key, value


def default_enter(path, key, value):
    if isinstance(value, Mapping):
        try:
            new_parent = value.__class__()
        except Exception:
            new_parent = {}
        return new_parent, ItemsView(value)

    if isinstance(value, (list, tuple)):
        return [], enumerate(value)

    if isinstance(value, Set) and not isinstance(value, (str, bytes)):
        try:
            new_parent = value.__class__()
        except Exception:
            new_parent = set()
        return new_parent, enumerate(value)

    return value, False


def remap(root, visit=default_visit, enter=default_enter, exit=None,
          reraise_visit=True, trace=()):
    if visit is None:
        visit = default_visit
    if enter is None:
        enter = default_enter

    if trace is True:
        trace = ('visit', 'enter', 'exit')
    elif isinstance(trace, str):
        trace = (trace,)
    else:
        trace = tuple(trace)

    registry = {}
    stack = [(None, root)]
    path_stack = []
    new_items_stack = []
    parent_stack = []

    while stack:
        key, value = stack.pop()

        if value is _REMAP_EXIT:
            key, old_parent, new_parent, new_items, path = parent_stack.pop()
            if exit is not None:
                try:
                    exit(path, key, old_parent, new_parent, new_items)
                except Exception:
                    if reraise_visit:
                        raise
            if isinstance(old_parent, tuple) and isinstance(new_parent, list):
                new_parent = old_parent.__class__(new_parent)
                registry[id(old_parent)] = new_parent
                if parent_stack:
                    parent_stack[-1][3][-1] = (key, new_parent)
                else:
                    root = new_parent
            continue

        if id(value) in registry:
            value = registry[id(value)]
            if parent_stack:
                parent_stack[-1][3].append((key, value))
            else:
                root = value
            continue

        path = tuple(path_stack)
        if key is not None:
            path = path + (key,)

        if 'enter' in trace:
            print(' .. remap enter:', path, '-', value)

        try:
            new_parent, new_items = enter(path, key, value)
        except Exception:
            if reraise_visit:
                raise
            new_parent, new_items = value, False

        if not new_items:
            registry[id(value)] = new_parent
            if parent_stack:
                parent_stack[-1][3].append((key, new_parent))
            else:
                root = new_parent
            continue

        registry[id(value)] = new_parent
        frame = [key, value, new_parent, [], path[:-1] if key is not None else ()]
        parent_stack.append(frame)
        stack.append((None, _REMAP_EXIT))

        items = list(new_items)
        for child_key, child_value in reversed(items):
            stack.append((child_key, child_value))

        while stack and stack[-1][1] is not _REMAP_EXIT:
            child_key, child_value = stack.pop()
            child_path = path
            if id(child_value) in registry:
                new_value = registry[id(child_value)]
            else:
                try:
                    child_parent, child_items = enter(child_path, child_key, child_value)
                except Exception:
                    if reraise_visit:
                        raise
                    child_parent, child_items = child_value, False

                if child_items:
                    stack.append((child_key, child_value))
                    break
                registry[id(child_value)] = child_parent
                new_value = child_parent

            try:
                visited = visit(child_path, child_key, new_value)
            except Exception:
                if reraise_visit:
                    raise
                visited = False

            if 'visit' in trace:
                print(' .. remap visit:', child_path, '-', child_key, '-', new_value)

            if visited is False or visited is None:
                continue
            if visited is True:
                new_key, new_value = child_key, new_value
            else:
                new_key, new_value = visited
            frame[3].append((new_key, new_value))

        if stack and stack[-1][1] is not _REMAP_EXIT:
            continue

        frame = parent_stack[-1]
        for item_key, item_value in frame[3]:
            if isinstance(frame[2], Mapping):
                frame[2][item_key] = item_value
            elif isinstance(frame[2], set):
                frame[2].add(item_value)
            else:
                frame[2].append(item_value)

        if 'exit' in trace:
            print(' .. remap exit:', path, '-', value)

        if parent_stack:
            parent_stack.pop()
        if isinstance(value, tuple) and isinstance(new_parent, list):
            new_parent = value.__class__(new_parent)
            registry[id(value)] = new_parent
        if parent_stack:
            parent_stack[-1][3].append((key, new_parent))
        else:
            root = new_parent

    return root


def research(root, query=lambda path, key, value: True, reraise=True):
    results = []

    def visit(path, key, value):
        try:
            if query(path, key, value):
                results.append((path + (key,), value))
        except Exception:
            if reraise:
                raise
        return key, value

    remap(root, visit=visit, reraise_visit=reraise)
    return results


class PathAccessError(Exception):
    def __init__(self, exc, segment, path):
        self.exc = exc
        self.segment = segment
        self.path = path
        message = "could not access %r from path %r, error: %s" % (
            segment, path, exc)
        super().__init__(message)


def get_path(root, path, default=_UNSET):
    if isinstance(path, str):
        path = path.split('.')
    else:
        path = list(path)

    current = root
    for segment in path:
        try:
            try:
                current = current[segment]
            except (KeyError, IndexError, TypeError):
                if isinstance(segment, str):
                    try:
                        current = current[int(segment)]
                    except (ValueError, KeyError, IndexError, TypeError):
                        current = getattr(current, segment)
                else:
                    raise
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            if default is not _UNSET:
                return default
            raise PathAccessError(exc, segment, path)
    return current


def backoff(start, stop, count=None, factor=2.0, jitter=False):
    return list(backoff_iter(start, stop, count, factor, jitter))


def backoff_iter(start, stop, count=None, factor=2.0, jitter=False):
    if start <= 0:
        raise ValueError('expected a positive start value')
    if stop < start:
        raise ValueError('expected stop to be greater than or equal to start')
    if factor <= 1:
        raise ValueError('expected factor to be greater than one')

    if count is None:
        count = int(math.ceil(math.log(stop / float(start), factor))) + 1
    if count < 0:
        raise ValueError('expected a non-negative count')

    current = float(start)
    for _ in range(count):
        value = min(current, stop)
        if jitter:
            if jitter is True:
                value = random.random() * value
            else:
                value = random.uniform(value * (1 - jitter), value * (1 + jitter))
        yield value
        current *= factor


def bucketize(src, key=bool, value_transform=None, key_filter=None):
    if key is None:
        key = lambda value: value
    if value_transform is None:
        value_transform = lambda value: value

    if key_filter is None:
        allows = None
    elif callable(key_filter):
        allows = key_filter
    else:
        accepted = frozenset(key_filter)
        allows = accepted.__contains__

    buckets = {}
    for value in src:
        bucket_key = key(value)
        if allows is not None and not allows(bucket_key):
            continue
        buckets.setdefault(bucket_key, []).append(value_transform(value))
    return buckets


def iter_len(iterable):
    try:
        return len(iterable)
    except TypeError:
        return sum(1 for _ in iterable)


def float_range(start, stop=None, step=None):
    if stop is None:
        start, stop = 0.0, start
    if step is None:
        step = 1.0

    start = float(start)
    stop = float(stop)
    step = float(step)

    if step == 0:
        raise ValueError('step must not be zero')

    current = start
    if step > 0:
        while current < stop:
            yield current
            current += step
    else:
        while current > stop:
            yield current
            current += step


def frange(start, stop=None, step=None):
    return list(float_range(start, stop, step))
# -*- coding: utf-8 -*-
"""Various utility classes and functions."""

import re

from genshi.compat import html_entities, unichr

__docformat__ = 'restructuredtext en'


class LRUCache(dict):

    class _Item(object):

        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.prv = None
            self.nxt = None

        def __repr__(self):
            return repr(self.value)

    def __init__(self, capacity):
        self._dict = {}
        self.capacity = capacity
        self.head = None
        self.tail = None

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        item = self.head
        while item is not None:
            yield item.key
            item = item.nxt

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        item = self._dict[key]
        self._update_item(item)
        return item.value

    def __setitem__(self, key, value):
        item = self._dict.get(key)
        if item is None:
            item = self._Item(key, value)
            self._dict[key] = item
            self._insert_item(item)
        else:
            item.value = value
            self._update_item(item)
            self._manage_size()

    def __repr__(self):
        return repr(self._dict)

    def _insert_item(self, item):
        item.prv = None
        item.nxt = self.head
        if self.head is None:
            self.tail = item
        else:
            self.head.prv = item
        self.head = item
        self._manage_size()

    def _manage_size(self):
        while len(self._dict) > self.capacity:
            old_tail = self.tail
            del self._dict[old_tail.key]
            if old_tail == self.head:
                self.head = None
                self.tail = None
            else:
                self.tail = old_tail.prv
                self.tail.nxt = None

    def _update_item(self, item):
        if item == self.head:
            return

        previous = item.prv
        following = item.nxt
        previous.nxt = following
        if following is None:
            self.tail = previous
        else:
            following.prv = previous

        item.prv = None
        item.nxt = self.head
        self.head.prv = item
        self.head = item


def flatten(items):
    result = []
    for item in items:
        if isinstance(item, (frozenset, list, set, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


_STRIPENTITIES_RE = re.compile(
    r'&(?:#((?:\d+)|(?:[xX][0-9a-fA-F]+));?|(\w+);)'
)


def stripentities(text, keepxmlentities=False):
    def replace(match):
        number = match.group(1)
        if number:
            if number.startswith('x'):
                number = int(number[1:], 16)
            else:
                number = int(number, 10)
            return unichr(number)

        name = match.group(2)
        if keepxmlentities and name in ('amp', 'apos', 'gt', 'lt', 'quot'):
            return '&%s;' % name
        try:
            return unichr(html_entities.name2codepoint[name])
        except KeyError:
            if keepxmlentities:
                return '&amp;%s;' % name
            return name

    return _STRIPENTITIES_RE.sub(replace, text)


_STRIPTAGS_RE = re.compile(r'(<!--.*?-->|<[^>]*>)')


def striptags(text):
    return _STRIPTAGS_RE.sub('', text)


def plaintext(text, keeplinebreaks=True):
    result = stripentities(striptags(text))
    if not keeplinebreaks:
        result = result.replace('\n', ' ')
    return result
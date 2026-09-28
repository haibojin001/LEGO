# -*- coding: utf-8 -*-
"""HTML stream filters."""

import re

from genshi.compat import text_type
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT

__all__ = ['HTMLFormFiller', 'HTMLSanitizer']
__docformat__ = 'restructuredtext en'


class HTMLFormFiller(object):

    def __init__(self, name=None, id=None, data=None, passwords=False):
        self.name = name
        self.id = id
        self.data = {} if data is None else data
        self.passwords = passwords

    def __call__(self, stream):
        in_form = False
        in_select = False
        in_option = False
        in_textarea = False

        select_value = None
        option_value = None
        textarea_value = None
        option_start = None
        option_text = []
        no_option_value = False

        for kind, data, pos in stream:
            if kind is START:
                tag, attrs = data
                tagname = tag.localname

                if tagname == 'form' and (
                    (self.name and attrs.get('name') == self.name) or
                    (self.id and attrs.get('id') == self.id) or
                    not (self.name or self.id)
                ):
                    in_form = True

                elif in_form:
                    if tagname == 'input':
                        input_type = attrs.get('type', '').lower()

                        if input_type in ('checkbox', 'radio'):
                            name = attrs.get('name')
                            if name and name in self.data:
                                field_value = self.data[name]
                                declared = attrs.get('value')
                                checked = False

                                if isinstance(field_value, (list, tuple)):
                                    if declared is not None:
                                        checked = declared in [
                                            text_type(value)
                                            for value in field_value
                                        ]
                                    else:
                                        checked = any(field_value)
                                else:
                                    if declared is not None:
                                        checked = declared == text_type(field_value)
                                    elif input_type == 'checkbox':
                                        checked = bool(field_value)

                                if checked:
                                    attrs |= [(QName('checked'), 'checked')]
                                elif 'checked' in attrs:
                                    attrs -= 'checked'

                        elif (input_type in ('', 'hidden', 'text') or
                              (input_type == 'password' and self.passwords)):
                            name = attrs.get('name')
                            if name and name in self.data:
                                field_value = self.data[name]
                                if isinstance(field_value, (list, tuple)):
                                    field_value = field_value[0]
                                if field_value is not None:
                                    attrs |= [
                                        (QName('value'), text_type(field_value))
                                    ]

                    elif tagname == 'select':
                        name = attrs.get('name')
                        if name in self.data:
                            select_value = self.data[name]
                            in_select = True

                    elif tagname == 'textarea':
                        name = attrs.get('name')
                        if name in self.data:
                            textarea_value = self.data.get(name)
                            if isinstance(textarea_value, (list, tuple)):
                                textarea_value = textarea_value[0]
                            in_textarea = True

                    elif in_select and tagname == 'option':
                        option_start = (kind, data, pos)
                        option_value = attrs.get('value')
                        if option_value is None:
                            no_option_value = True
                            option_value = ''
                        in_option = True
                        continue

                yield kind, (tag, attrs), pos

            elif in_form and kind is TEXT:
                if in_select and in_option:
                    if no_option_value:
                        option_value += data
                    option_text.append((kind, data, pos))
                    continue
                if in_textarea:
                    continue
                yield kind, data, pos

            elif in_form and kind is END:
                tagname = data.localname

                if tagname == 'form':
                    in_form = False

                elif tagname == 'select':
                    in_select = False
                    select_value = None

                elif in_select and tagname == 'option':
                    if isinstance(select_value, (list, tuple)):
                        selected = option_value in [
                            text_type(value) for value in select_value
                        ]
                    else:
                        selected = option_value == text_type(select_value)

                    start_kind, (tag, attrs), start_pos = option_start
                    if selected:
                        attrs |= [(QName('selected'), 'selected')]
                    elif 'selected' in attrs:
                        attrs -= 'selected'

                    yield start_kind, (tag, attrs), start_pos
                    for event in option_text:
                        yield event

                    in_option = False
                    no_option_value = False
                    option_start = None
                    option_value = None
                    option_text = []

                elif in_textarea and tagname == 'textarea':
                    if textarea_value:
                        yield TEXT, text_type(textarea_value), pos
                        textarea_value = None
                    in_textarea = False

                yield kind, data, pos

            else:
                yield kind, data, pos


class HTMLSanitizer(object):

    SAFE_TAGS = frozenset([
        'a', 'abbr', 'acronym', 'address', 'area', 'b', 'big', 'blockquote',
        'br', 'button', 'caption', 'center', 'cite', 'code', 'col',
        'colgroup', 'dd', 'del', 'div', 'dfn', 'dir', 'dl', 'dt', 'em',
        'fieldset', 'font', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'hr', 'i', 'img', 'input', 'ins', 'kbd', 'label', 'legend', 'li',
        'map', 'menu', 'ol', 'optgroup', 'option', 'p', 'pre', 'q', 's',
        'samp', 'select', 'small', 'span', 'strike', 'strong', 'sub', 'sup',
        'table', 'tbody', 'td', 'textarea', 'tfoot', 'th', 'thead', 'tr',
        'tt', 'u', 'ul', 'var'
    ])

    SAFE_ATTRS = frozenset([
        'abbr', 'accept', 'accept-charset', 'accesskey', 'action', 'align',
        'alt', 'archive', 'axis', 'background', 'border', 'cellpadding',
        'cellspacing', 'char', 'charoff', 'charset', 'checked', 'cite',
        'class', 'classid', 'clear', 'codebase', 'codetype', 'color', 'cols',
        'colspan', 'compact', 'contenteditable', 'coords', 'datetime', 'dir',
        'disabled', 'enctype', 'face', 'for', 'frame', 'headers', 'height',
        'href', 'hreflang', 'hspace', 'id', 'label', 'lang', 'longdesc',
        'maxlength', 'media', 'method', 'multiple', 'name', 'nohref',
        'noshade', 'nowrap', 'prompt', 'readonly', 'rel', 'rev', 'rows',
        'rowspan', 'rules', 'scope', 'selected', 'shape', 'size', 'span',
        'src', 'start', 'summary', 'tabindex', 'target', 'title', 'type',
        'usemap', 'valign', 'value', 'vspace', 'width'
    ])

    SAFE_SCHEMES = frozenset([
        'file', 'ftp', 'http', 'https', 'mailto', None
    ])

    URI_ATTRS = frozenset([
        'action', 'archive', 'background', 'cite', 'classid', 'codebase',
        'data', 'dynsrc', 'href', 'longdesc', 'profile', 'src', 'usemap'
    ])

    _EXPRESSION_SEARCH = re.compile(
        r'''
            expression\s*\(
            | \bbehavior\s*:
            | \bbehaviour\s*:
            | -\s*moz\s*-\s*binding\s*:
            | \bmoz-binding\s*:
        ''',
        re.IGNORECASE | re.VERBOSE
    ).search

    _URL_FINDITER = re.compile(
        r'''
            url\s*\(\s*
            (?:
                "([^"]*)"
                | '([^']*)'
                | ([^)]*)
            )
            \s*\)
        ''',
        re.IGNORECASE | re.VERBOSE
    ).finditer

    _COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)

    def __init__(self, safe_tags=SAFE_TAGS, safe_attrs=SAFE_ATTRS,
                 safe_schemes=SAFE_SCHEMES):
        self.safe_tags = safe_tags
        self.safe_attrs = safe_attrs
        self.safe_schemes = safe_schemes

    def is_safe_uri(self, uri):
        if uri is None:
            return True

        uri = stripentities(uri).lower()
        uri = ''.join(char for char in uri if char not in '\t\r\n\f ')
        if ':' not in uri:
            return True

        scheme = uri.split(':', 1)[0]
        return scheme in self.safe_schemes

    def sanitize_css(self, text):
        text = self._COMMENT_RE.sub('', text)
        declarations = []

        for declaration in text.split(';'):
            if self._EXPRESSION_SEARCH(declaration):
                continue

            unsafe = False
            for match in self._URL_FINDITER(declaration):
                uri = match.group(1)
                if uri is None:
                    uri = match.group(2)
                if uri is None:
                    uri = match.group(3)
                if not self.is_safe_uri(uri.strip()):
                    unsafe = True
                    break

            if not unsafe:
                declarations.append(declaration)

        return ';'.join(declarations)

    def __call__(self, stream):
        waiting_for = None

        for kind, data, pos in stream:
            if kind is START:
                tag, attrs = data
                tagname = tag.localname

                if tagname not in self.safe_tags:
                    if tagname in ('script', 'style'):
                        waiting_for = tag
                    continue

                if waiting_for is not None:
                    continue

                clean_attrs = []
                for attr, value in attrs:
                    attrname = attr.localname
                    if attrname not in self.safe_attrs:
                        continue

                    if attrname in self.URI_ATTRS:
                        if not self.is_safe_uri(value):
                            continue
                    elif attrname == 'style':
                        value = self.sanitize_css(value)

                    clean_attrs.append((attr, value))

                yield kind, (tag, Attrs(clean_attrs)), pos

            elif kind is END:
                if waiting_for is not None:
                    if data == waiting_for:
                        waiting_for = None
                    continue

                if data.localname in self.safe_tags:
                    yield kind, data, pos

            elif waiting_for is None:
                yield kind, data, pos
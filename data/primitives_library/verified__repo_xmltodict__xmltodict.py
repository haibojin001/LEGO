#!/usr/bin/env python
"""Makes working with XML feel like you are working with JSON."""

import codecs
import re
from collections.abc import Iterable
from inspect import isgenerator
from io import StringIO
from xml.parsers import expat
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl


__version__ = "0.14.2"


_XML_NAME_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_.-]*:)?[A-Za-z_][A-Za-z0-9_.-]*$"
)


class ParsingInterrupted(Exception):
    pass


class _DictSAXHandler:
    def __init__(
        self,
        item_depth=0,
        item_callback=lambda *args: True,
        xml_attribs=True,
        attr_prefix="@",
        cdata_key="#text",
        force_cdata=False,
        cdata_separator="",
        postprocessor=None,
        dict_constructor=dict,
        strip_whitespace=True,
        namespace_separator=":",
        namespaces=None,
        force_list=None,
        comment_key="#comment",
    ):
        self.path = []
        self.stack = []
        self.data = []
        self.item = None
        self.item_depth = item_depth
        self.item_callback = item_callback
        self.xml_attribs = xml_attribs
        self.attr_prefix = attr_prefix
        self.cdata_key = cdata_key
        self.force_cdata = force_cdata
        self.cdata_separator = cdata_separator
        self.postprocessor = postprocessor
        self.dict_constructor = dict_constructor
        self.strip_whitespace = strip_whitespace
        self.namespace_separator = namespace_separator
        self.namespaces = namespaces
        self.namespace_declarations = dict_constructor()
        self.force_list = force_list
        self.comment_key = comment_key

    def _build_name(self, full_name):
        if self.namespaces is None:
            return full_name

        position = full_name.rfind(self.namespace_separator)
        if position < 0:
            return full_name

        namespace = full_name[:position]
        name = full_name[position + 1:]

        try:
            namespace = self.namespaces[namespace]
        except KeyError:
            pass

        if not namespace:
            return name
        return self.namespace_separator.join((namespace, name))

    def _attrs_to_dict(self, attrs):
        if isinstance(attrs, dict):
            return attrs
        return self.dict_constructor(zip(attrs[0::2], attrs[1::2]))

    def startNamespaceDecl(self, prefix, uri):
        self.namespace_declarations[prefix or ""] = uri

    def startElement(self, full_name, attrs):
        name = self._build_name(full_name)
        attrs = self._attrs_to_dict(attrs)

        if self.namespace_declarations:
            if not attrs:
                attrs = self.dict_constructor()
            attrs["xmlns"] = self.namespace_declarations
            self.namespace_declarations = self.dict_constructor()

        self.path.append((name, attrs or None))

        if len(self.path) >= self.item_depth:
            self.stack.append((self.item, self.data))

            if self.xml_attribs:
                entries = []
                for key, value in attrs.items():
                    key = self.attr_prefix + self._build_name(key)
                    if self.postprocessor is not None:
                        entry = self.postprocessor(self.path, key, value)
                    else:
                        entry = (key, value)
                    if entry:
                        entries.append(entry)
                attrs = self.dict_constructor(entries)
            else:
                attrs = None

            self.item = attrs or None
            self.data = []

    def endElement(self, full_name):
        name = self._build_name(full_name)

        if len(self.path) == self.item_depth:
            text = None if not self.data else self.cdata_separator.join(self.data)
            item = self.item

            if self.strip_whitespace and text:
                text = text.strip() or None

            if text and self._should_force_cdata(name, text) and item is None:
                item = self.dict_constructor()

            if item is not None:
                if text:
                    self.push_data(item, self.cdata_key, text)
            else:
                item = text

            if not self.item_callback(self.path, item):
                raise ParsingInterrupted

            if self.stack:
                self.item, self.data = self.stack.pop()
            else:
                self.item = None
                self.data = []

            self.path.pop()
            return

        if self.stack:
            text = None if not self.data else self.cdata_separator.join(self.data)
            item = self.item
            self.item, self.data = self.stack.pop()

            if self.strip_whitespace and text:
                text = text.strip() or None

            if text and self._should_force_cdata(name, text) and item is None:
                item = self.dict_constructor()

            if item is not None:
                if text:
                    self.push_data(item, self.cdata_key, text)
                self.item = self.push_data(self.item, name, item)
            else:
                self.item = self.push_data(self.item, name, text)
        else:
            self.item = None
            self.data = []

        self.path.pop()

    def characters(self, data):
        if self.data:
            self.data.append(data)
        else:
            self.data = [data]

    def comments(self, data):
        if self.strip_whitespace:
            data = data.strip()
        self.item = self.push_data(self.item, self.comment_key, data)

    def push_data(self, item, key, data):
        if self.postprocessor is not None:
            result = self.postprocessor(self.path, key, data)
            if result is None:
                return item
            key, data = result

        if item is None:
            item = self.dict_constructor()

        try:
            existing = item[key]
        except KeyError:
            if self._should_force_list(key, data):
                item[key] = [data]
            else:
                item[key] = data
        else:
            if isinstance(existing, list):
                existing.append(data)
            else:
                item[key] = [existing, data]

        return item

    def _should_force_list(self, key, value):
        if not self.force_list:
            return False
        if isinstance(self.force_list, bool):
            return self.force_list
        try:
            return key in self.force_list
        except TypeError:
            return self.force_list(self.path[:-1], key, value)

    def _should_force_cdata(self, key, value):
        if not self.force_cdata:
            return False
        if isinstance(self.force_cdata, bool):
            return self.force_cdata
        try:
            return key in self.force_cdata
        except TypeError:
            return self.force_cdata(self.path[:-1], key, value)


def parse(
    xml_input,
    encoding=None,
    expat=expat,
    process_namespaces=False,
    namespace_separator=":",
    disable_entities=True,
    process_comments=False,
    **kwargs
):
    handler = _DictSAXHandler(namespace_separator=namespace_separator, **kwargs)

    if process_namespaces:
        parser = expat.ParserCreate(encoding, namespace_separator)
        parser.StartNamespaceDeclHandler = handler.startNamespaceDecl
    else:
        parser = expat.ParserCreate(encoding)

    try:
        parser.ordered_attributes = True
    except AttributeError:
        pass

    parser.StartElementHandler = handler.startElement
    parser.EndElementHandler = handler.endElement
    parser.CharacterDataHandler = handler.characters

    if process_comments:
        parser.CommentHandler = handler.comments

    if disable_entities:
        parser.DefaultHandler = lambda value: None
        parser.ExternalEntityRefHandler = lambda *args: 1

    try:
        if isinstance(xml_input, (str, bytes)):
            parser.Parse(xml_input, True)
        elif hasattr(xml_input, "read"):
            while True:
                chunk = xml_input.read(1024 * 1024)
                if not chunk:
                    break
                parser.Parse(chunk, False)
            parser.Parse(b"", True)
        elif isgenerator(xml_input) or (
            isinstance(xml_input, Iterable) and not isinstance(xml_input, (str, bytes))
        ):
            for chunk in xml_input:
                parser.Parse(chunk, False)
            parser.Parse(b"", True)
        else:
            parser.Parse(xml_input, True)
    except ParsingInterrupted:
        pass

    return handler.item


def _process_namespace(name, namespaces, namespace_separator=":", attr_prefix="@"):
    if not namespaces:
        return name

    attribute_prefix = ""
    if name.startswith(attr_prefix):
        attribute_prefix = attr_prefix
        name = name[len(attr_prefix):]

    try:
        namespace, local_name = name.split(namespace_separator, 1)
    except ValueError:
        return attribute_prefix + name

    return attribute_prefix + namespace_separator.join(
        (namespaces.get(namespace, namespace), local_name)
    )


def _validate_name(name):
    if not isinstance(name, str) or not _XML_NAME_RE.match(name):
        raise ValueError("Invalid XML name: %r" % name)
    return name


def _stringify(value, encoding="utf-8", errors="replace"):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(encoding, errors)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _emit(
    key,
    value,
    content_handler,
    attr_prefix="@",
    cdata_key="#text",
    depth=0,
    preprocessor=None,
    pretty=False,
    newl="\n",
    indent="\t",
    namespace_separator=":",
    namespaces=None,
    full_document=True,
    short_empty_elements=False,
    comment_key="#comment",
    encoding="utf-8",
    errors="replace",
    **kwargs
):
    key = _process_namespace(
        key,
        namespaces,
        namespace_separator=namespace_separator,
        attr_prefix=attr_prefix,
    )

    if preprocessor is not None:
        result = preprocessor(key, value)
        if result is None:
            return
        key, value = result
        key = _process_namespace(
            key,
            namespaces,
            namespace_separator=namespace_separator,
            attr_prefix=attr_prefix,
        )

    if key == comment_key:
        if isinstance(value, (list, tuple)):
            for item in value:
                _emit(
                    key,
                    item,
                    content_handler,
                    attr_prefix=attr_prefix,
                    cdata_key=cdata_key,
                    depth=depth,
                    preprocessor=preprocessor,
                    pretty=pretty,
                    newl=newl,
                    indent=indent,
                    namespace_separator=namespace_separator,
                    namespaces=namespaces,
                    full_document=full_document,
                    short_empty_elements=short_empty_elements,
                    comment_key=comment_key,
                    encoding=encoding,
                    errors=errors,
                    **kwargs
                )
            return
        if pretty:
            content_handler.ignorableWhitespace(indent * depth)
        comment = _stringify(value, encoding=encoding, errors=errors)
        if "--" in comment or comment.endswith("-"):
            raise ValueError("Invalid XML comment")
        content_handler._write("<!--%s-->" % comment)
        if pretty:
            content_handler.ignorableWhitespace(newl)
        return

    _validate_name(key)

    if not isinstance(value, (list, tuple)):
        values = [value]
    else:
        values = value

    for index, value in enumerate(values):
        if full_document and depth == 0 and index > 0:
            raise ValueError("Document must have exactly one root.")

        attrs = {}
        cdata = None
        children = []

        if isinstance(value, dict):
            for item_key, item_value in value.items():
                if item_key == cdata_key:
                    cdata = item_value
                elif item_key.startswith(attr_prefix):
                    attr_name = _process_namespace(
                        item_key[len(attr_prefix):],
                        namespaces,
                        namespace_separator=namespace_separator,
                        attr_prefix=attr_prefix,
                    )
                    _validate_name(attr_name)
                    if item_value is not None:
                        attrs[attr_name] = _stringify(
                            item_value, encoding=encoding, errors=errors
                        )
                else:
                    children.append((item_key, item_value))
        elif value is not None:
            cdata = value

        if pretty:
            content_handler.ignorableWhitespace(indent * depth)

        content_handler.startElement(key, AttributesImpl(attrs))

        has_children = False
        for child_key, child_value in children:
            if isinstance(child_value, (list, tuple)) and not child_value:
                continue
            has_children = True
            if pretty:
                content_handler.ignorableWhitespace(newl)
            _emit(
                child_key,
                child_value,
                content_handler,
                attr_prefix=attr_prefix,
                cdata_key=cdata_key,
                depth=depth + 1,
                preprocessor=preprocessor,
                pretty=pretty,
                newl=newl,
                indent=indent,
                namespace_separator=namespace_separator,
                namespaces=namespaces,
                full_document=full_document,
                short_empty_elements=short_empty_elements,
                comment_key=comment_key,
                encoding=encoding,
                errors=errors,
                **kwargs
            )

        if cdata is not None:
            content_handler.characters(
                _stringify(cdata, encoding=encoding, errors=errors)
            )

        if pretty and has_children:
            content_handler.ignorableWhitespace(indent * depth)

        content_handler.endElement(key)

        if pretty:
            content_handler.ignorableWhitespace(newl)


def unparse(
    input_dict,
    output=None,
    encoding="utf-8",
    full_document=True,
    short_empty_elements=False,
    **kwargs
):
    if not isinstance(input_dict, dict):
        raise ValueError("unparse must receive a dict")

    if len(input_dict) != 1:
        raise ValueError("Document must have exactly one root.")

    if output is None:
        output = StringIO()
        return_output = True
    else:
        return_output = False

    content_handler = XMLGenerator(
        output,
        encoding,
        short_empty_elements=short_empty_elements,
    )

    if full_document:
        content_handler.startDocument()

    root_key, root_value = next(iter(input_dict.items()))
    _emit(
        root_key,
        root_value,
        content_handler,
        full_document=full_document,
        short_empty_elements=short_empty_elements,
        encoding=encoding,
        **kwargs
    )

    if full_document:
        content_handler.endDocument()

    if return_output:
        return output.getvalue()
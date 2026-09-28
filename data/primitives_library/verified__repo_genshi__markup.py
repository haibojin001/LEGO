from itertools import chain

from genshi.core import Attrs, Markup, Namespace, Stream
from genshi.core import START, END, START_NS, END_NS, TEXT, PI, COMMENT
from genshi.input import XMLParser
from genshi.template.base import (
    BadDirectiveError, Template, TemplateSyntaxError, _apply_directives,
    EXEC, INCLUDE, SUB
)
from genshi.template.eval import Suite
from genshi.template.interpolation import interpolate
from genshi.template.directives import *
from genshi.template.text import NewTextTemplate


__all__ = ['MarkupTemplate']
__docformat__ = 'restructuredtext en'


class MarkupTemplate(Template):

    DIRECTIVE_NAMESPACE = 'http://genshi.edgewall.org/'
    XINCLUDE_NAMESPACE = 'http://www.w3.org/2001/XInclude'

    directives = [
        ('def', DefDirective),
        ('match', MatchDirective),
        ('when', WhenDirective),
        ('otherwise', OtherwiseDirective),
        ('for', ForDirective),
        ('if', IfDirective),
        ('choose', ChooseDirective),
        ('with', WithDirective),
        ('replace', ReplaceDirective),
        ('content', ContentDirective),
        ('attrs', AttrsDirective),
        ('strip', StripDirective),
    ]

    serializer = 'xml'
    _number_conv = Markup

    def __init__(self, source, filepath=None, filename=None, loader=None,
                 encoding=None, lookup='strict', allow_exec=True):
        Template.__init__(
            self, source, filepath=filepath, filename=filename,
            loader=loader, encoding=encoding, lookup=lookup,
            allow_exec=allow_exec
        )
        self.add_directives(self.DIRECTIVE_NAMESPACE, self)

    def _init_filters(self):
        Template._init_filters(self)
        self.filters.remove(self._include)
        self.filters += [self._match, self._include]

    def _parse(self, source, encoding):
        if not isinstance(source, Stream):
            source = XMLParser(source, filename=self.filename,
                               encoding=encoding)

        output = []
        for kind, data, pos in source:
            if kind is TEXT:
                for event_kind, event_data, event_pos in interpolate(
                        data, self.filepath, pos[1], pos[2],
                        lookup=self.lookup):
                    output.append((event_kind, event_data, event_pos))

            elif kind is PI and data[0] == 'python':
                if not self.allow_exec:
                    raise TemplateSyntaxError(
                        'Python code blocks not allowed',
                        self.filepath, *pos[1:]
                    )
                try:
                    code = Suite(data[1], self.filepath, pos[1],
                                 lookup=self.lookup)
                except SyntaxError as err:
                    raise TemplateSyntaxError(
                        err, self.filepath,
                        pos[1] + (err.lineno or 1) - 1,
                        pos[2] + (err.offset or 0)
                    )
                output.append((EXEC, code, pos))

            elif kind is COMMENT:
                if not data.lstrip().startswith('!'):
                    output.append((kind, data, pos))

            else:
                output.append((kind, data, pos))

        return output

    def _prepare(self, stream):
        stream = self._extract_directives(
            stream, self.DIRECTIVE_NAMESPACE, self
        )
        return self._extract_includes(stream)

    def _extract_directives(self, stream, namespace, factory):
        depth = 0
        pending = {}
        result = []
        namespace_prefixes = {}

        for kind, data, pos in stream:
            if kind is START:
                tag, attrs = data
                directives = []
                strip_tag = False

                if tag.namespace == namespace:
                    directive_class = factory.get_directive(tag.localname)
                    if directive_class is None:
                        raise BadDirectiveError(
                            tag.localname, self.filepath, pos[1]
                        )

                    arguments = dict(
                        (name.localname, value)
                        for name, value in attrs
                        if not name.namespace
                    )
                    directives.append((
                        factory.get_directive_index(directive_class),
                        directive_class, arguments,
                        namespace_prefixes.copy(), pos
                    ))
                    strip_tag = True

                retained_attrs = []
                for name, value in attrs:
                    if name.namespace == namespace:
                        directive_class = factory.get_directive(name.localname)
                        if directive_class is None:
                            raise BadDirectiveError(
                                name.localname, self.filepath, pos[1]
                            )

                        if type(value) is list and len(value) == 1:
                            value = value[0][1]

                        directives.append((
                            factory.get_directive_index(directive_class),
                            directive_class, value,
                            namespace_prefixes.copy(), pos
                        ))
                    else:
                        retained_attrs.append((name, value))

                if directives:
                    directives.sort(key=lambda item: item[0])
                    pending[(depth, tag)] = (
                        directives, len(result), strip_tag
                    )

                result.append((START, (tag, Attrs(retained_attrs)), pos))
                depth += 1

            elif kind is END:
                depth -= 1
                result.append((kind, data, pos))

                entry = pending.pop((depth, data), None)
                if entry is not None:
                    directives, start_index, strip_tag = entry
                    program = result[start_index:]
                    if strip_tag:
                        program = program[1:-1]
                    result[start_index:] = [
                        (SUB, (directives, program), pos)
                    ]

            elif kind is SUB:
                directives, program = data
                extracted = self._extract_directives(
                    program, namespace, factory
                )

                if (len(extracted) == 1 and extracted[0][0] is SUB and
                        (program[0][0] is not SUB or
                         program[0][1][0] != extracted[0][1][0])):
                    extra_directives, extracted = extracted[0][1]
                    directives += extra_directives

                result.append((SUB, (directives, extracted), pos))

            elif kind is START_NS:
                prefix, uri = data
                namespace_prefixes[prefix] = uri
                if uri != namespace:
                    result.append((kind, data, pos))

            elif kind is END_NS:
                uri = namespace_prefixes.pop(data, None)
                if uri and uri != namespace:
                    result.append((kind, data, pos))

            else:
                result.append((kind, data, pos))

        return result

    def _extract_includes(self, stream):
        streams = [[]]
        prefixes = {}
        include_stack = []
        fallback_stack = []
        xinclude = Namespace(self.XINCLUDE_NAMESPACE)

        for kind, data, pos in stream:
            current = streams[-1]

            if kind is SUB:
                directives, substream = data
                current.append((
                    SUB,
                    (directives, self._extract_includes(substream)),
                    pos
                ))
                continue

            if kind is START_NS:
                prefix, uri = data
                prefixes[prefix] = uri
                if uri != self.XINCLUDE_NAMESPACE:
                    current.append((kind, data, pos))
                continue

            if kind is END_NS:
                uri = prefixes.pop(data, None)
                if uri != self.XINCLUDE_NAMESPACE:
                    current.append((kind, data, pos))
                continue

            if kind is START:
                tag, attrs = data
                if tag in xinclude:
                    if tag.localname == 'include':
                        href = attrs.get('href')
                        if not href:
                            raise TemplateSyntaxError(
                                'Include misses required "href" attribute',
                                self.filepath, *pos[1:]
                            )

                        parse = attrs.get('parse', 'xml')
                        if parse not in ('xml', 'text'):
                            raise TemplateSyntaxError(
                                'Invalid value "%s" for "parse" attribute'
                                % parse,
                                self.filepath, *pos[1:]
                            )

                        include_stack.append({
                            'href': href,
                            'class': (MarkupTemplate if parse == 'xml'
                                      else NewTextTemplate),
                            'pos': pos,
                            'fallback': None,
                        })
                        streams.append([])

                    elif tag.localname == 'fallback':
                        if not include_stack:
                            raise TemplateSyntaxError(
                                'xi:fallback element nested outside '
                                'xi:include',
                                self.filepath, *pos[1:]
                            )
                        fallback_stack.append(len(include_stack))
                        streams.append([])

                    else:
                        current.append((kind, data, pos))
                else:
                    current.append((kind, data, pos))
                continue

            if kind is END:
                tag = data
                if tag in xinclude:
                    if tag.localname == 'fallback':
                        if (not fallback_stack or
                                fallback_stack[-1] != len(include_stack)):
                            raise TemplateSyntaxError(
                                'xi:fallback element nested outside '
                                'xi:include',
                                self.filepath, *pos[1:]
                            )
                        fallback_stack.pop()
                        fallback = streams.pop()
                        include_stack[-1]['fallback'] = fallback

                    elif tag.localname == 'include':
                        if not include_stack:
                            current.append((kind, data, pos))
                            continue

                        streams.pop()
                        include = include_stack.pop()
                        streams[-1].append((
                            INCLUDE,
                            (include['href'], include['class'],
                             include['fallback']),
                            include['pos']
                        ))

                    else:
                        current.append((kind, data, pos))
                else:
                    current.append((kind, data, pos))
                continue

            current.append((kind, data, pos))

        return streams[0]
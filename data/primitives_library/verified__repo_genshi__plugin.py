# -*- coding: utf-8 -*-
"""Plugin API support for Genshi template engines."""

try:
    from importlib.resources import as_file as resources_as_file
    from importlib.resources import files as resources_files
except ImportError:
    from importlib_resources import as_file as resources_as_file
    from importlib_resources import files as resources_files

from genshi.compat import string_types
from genshi.input import ET, HTML, XML
from genshi.output import DocType
from genshi.template.base import Template
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
from genshi.template.text import NewTextTemplate, TextTemplate

__all__ = [
    'ConfigurationError',
    'AbstractTemplateEnginePlugin',
    'MarkupTemplateEnginePlugin',
    'TextTemplateEnginePlugin',
]

__docformat__ = 'restructuredtext en'


class ConfigurationError(ValueError):
    """Raised when template plugin configuration is invalid."""


class AbstractTemplateEnginePlugin(object):
    """Base implementation shared by the template engine plugins."""

    template_class = None
    extension = None

    def __init__(self, extra_vars_func=None, options=None):
        self.get_extra_vars = extra_vars_func
        if options is None:
            options = {}
        self.options = options

        self.default_encoding = options.get('genshi.default_encoding', None)

        auto_reload = options.get('genshi.auto_reload', '1')
        if isinstance(auto_reload, string_types):
            auto_reload = auto_reload.lower() in ('1', 'on', 'yes', 'true')

        configured_path = options.get('genshi.search_path', '')
        search_path = [item for item in configured_path.split(':') if item]
        self.use_package_naming = not search_path

        try:
            max_cache_size = int(options.get('genshi.max_cache_size', 25))
        except ValueError:
            raise ConfigurationError(
                'Invalid value for max_cache_size: "%s"' %
                options.get('genshi.max_cache_size')
            )

        callback = options.get('genshi.loader_callback', None)
        if callback and not hasattr(callback, '__call__'):
            raise ConfigurationError('loader callback must be a function')

        lookup_mode = options.get('genshi.lookup_errors', 'strict')
        if lookup_mode not in ('lenient', 'strict'):
            raise ConfigurationError(
                'Unknown lookup errors mode "%s"' % lookup_mode
            )

        try:
            allow_exec = bool(options.get('genshi.allow_exec', True))
        except ValueError:
            raise ConfigurationError(
                'Invalid value for allow_exec "%s"' %
                options.get('genshi.allow_exec')
            )

        self.loader = TemplateLoader(
            search_path,
            auto_reload=auto_reload,
            max_cache_size=max_cache_size,
            default_class=self.template_class,
            variable_lookup=lookup_mode,
            allow_exec=allow_exec,
            callback=callback,
        )

    def load_template(self, templatename, template_string=None):
        """Load a named template, or create a template from source text."""
        if template_string is not None:
            return self.template_class(template_string)

        if self.use_package_naming:
            split_at = templatename.rfind('.')
            if split_at >= 0:
                package_name = templatename[:split_at]
                filename = templatename[split_at + 1:] + self.extension
                resource = resources_files(package_name) / filename
                with resources_as_file(resource) as path:
                    return self.loader.load(str(path))

        return self.loader.load(templatename)

    def _get_render_options(self, format=None, fragment=False):
        if format is None:
            format = self.default_format

        options = {'method': format}
        if self.default_encoding:
            options['encoding'] = self.default_encoding
        return options

    def render(self, info, format=None, fragment=False, template=None):
        """Render a template with the supplied context."""
        options = self._get_render_options(format=format, fragment=fragment)
        return self.transform(info, template).render(**options)

    def transform(self, info, template):
        """Generate the event stream for a template and context."""
        if not isinstance(template, Template):
            template = self.load_template(template)
        return template.generate(**info)


class MarkupTemplateEnginePlugin(AbstractTemplateEnginePlugin):
    """Plugin implementation for markup templates."""

    template_class = MarkupTemplate
    extension = '.html'

    def __init__(self, extra_vars_func=None, options=None):
        AbstractTemplateEnginePlugin.__init__(self, extra_vars_func, options)

        configured_doctype = self.options.get('genshi.default_doctype')
        if configured_doctype:
            doctype = DocType.get(configured_doctype)
            if doctype is None:
                raise ConfigurationError(
                    'Unknown doctype %r' % configured_doctype
                )
            self.default_doctype = doctype
        else:
            self.default_doctype = None

        output_format = self.options.get(
            'genshi.default_format', 'html'
        ).lower()
        if output_format not in ('html', 'xhtml', 'xml', 'text'):
            raise ConfigurationError(
                'Unknown output format %r' % output_format
            )
        self.default_format = output_format

    def _get_render_options(self, format=None, fragment=False):
        options = super(MarkupTemplateEnginePlugin, self)._get_render_options(
            format, fragment
        )
        if self.default_doctype and not fragment:
            options['doctype'] = self.default_doctype
        return options

    def transform(self, info, template):
        """Generate markup template events with standard markup helpers."""
        context = {'ET': ET, 'HTML': HTML, 'XML': XML}
        if self.get_extra_vars:
            context.update(self.get_extra_vars())
        context.update(info)
        return super(MarkupTemplateEnginePlugin, self).transform(
            context, template
        )


class TextTemplateEnginePlugin(AbstractTemplateEnginePlugin):
    """Plugin implementation for text templates."""

    template_class = TextTemplate
    extension = '.txt'
    default_format = 'text'

    def __init__(self, extra_vars_func=None, options=None):
        if options is None:
            options = {}

        use_new_syntax = options.get('genshi.new_text_syntax')
        if isinstance(use_new_syntax, string_types):
            use_new_syntax = use_new_syntax.lower() in (
                '1', 'on', 'yes', 'true'
            )
        if use_new_syntax:
            self.template_class = NewTextTemplate

        AbstractTemplateEnginePlugin.__init__(self, extra_vars_func, options)
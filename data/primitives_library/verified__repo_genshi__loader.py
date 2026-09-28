try:
    from importlib.resources import open_binary as resources_open_binary
except ImportError:
    from importlib_resources import open_binary as resources_open_binary

import os

try:
    import threading
except ImportError:
    import dummy_threading as threading

from genshi.compat import string_types
from genshi.template.base import TemplateError
from genshi.util import LRUCache

__all__ = ['TemplateLoader', 'TemplateNotFound', 'directory', 'package',
           'prefixed']
__docformat__ = 'restructuredtext en'


class TemplateNotFound(TemplateError):
    """Raised when a requested template cannot be located."""

    def __init__(self, name, search_path):
        TemplateError.__init__(self, 'Template "%s" not found' % name)
        self.search_path = search_path


def directory(path):
    """Create a template loading function for a filesystem directory."""

    def load_template(filename):
        filepath = os.path.join(path, filename)
        if not os.path.isfile(filepath):
            raise TemplateNotFound(filename, [path])
        fileobj = open(filepath, 'rb')
        mtime = os.path.getmtime(filepath)

        def uptodate():
            return mtime == os.path.getmtime(filepath)

        return fileobj, filepath, uptodate

    return load_template


def package(name, path):
    """Create a template loading function for a package resource directory."""

    def load_template(filename):
        filepath = os.path.join(path, filename)
        try:
            return resources_open_binary(name, filepath), filepath, None
        except IOError:
            raise TemplateNotFound(filename, [path])

    return load_template


def prefixed(**delegates):
    """Create a loader that dispatches template names by their first path part."""

    def load_template(filename):
        parts = filename.split('/', 1)
        prefix = parts[0]
        if prefix not in delegates or len(parts) == 1:
            raise TemplateNotFound(filename, delegates.keys())
        return delegates[prefix](parts[1])

    return load_template


class TemplateLoader(object):
    """Loads, initializes, and caches templates found on a search path."""

    def __init__(self, search_path=None, auto_reload=False,
                 default_encoding=None, max_cache_size=25, default_class=None,
                 variable_lookup='strict', allow_exec=True, callback=None):
        from genshi.template.markup import MarkupTemplate

        self.search_path = search_path
        if self.search_path is None:
            self.search_path = []
        elif not isinstance(self.search_path, (list, tuple)):
            self.search_path = [self.search_path]

        self.auto_reload = auto_reload
        self.default_encoding = default_encoding
        self.default_class = default_class or MarkupTemplate
        self.variable_lookup = variable_lookup
        self.allow_exec = allow_exec
        if callback is not None and not hasattr(callback, '__call__'):
            raise TypeError('The "callback" parameter needs to be callable')
        self.callback = callback
        self._cache = LRUCache(max_cache_size)
        self._uptodate = {}
        self._lock = threading.RLock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_lock'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._lock = threading.RLock()

    def load(self, filename, relative_to=None, cls=None, encoding=None):
        if cls is None:
            cls = self.default_class

        search_path = self.search_path

        if relative_to and (not search_path or not os.path.isabs(relative_to)):
            filename = os.path.join(os.path.dirname(relative_to), filename)

        filename = os.path.normpath(filename)
        cachekey = filename

        self._lock.acquire()
        try:
            try:
                template = self._cache[cachekey]
                if not self.auto_reload:
                    return template
                uptodate = self._uptodate[cachekey]
                if uptodate is not None and uptodate():
                    return template
            except (KeyError, OSError):
                pass

            isabs = False
            if os.path.isabs(filename):
                search_path = [os.path.dirname(filename)]
                isabs = True
            elif relative_to and os.path.isabs(relative_to):
                search_path = [os.path.dirname(relative_to)] + list(search_path)
                isabs = True

            fileobj = None
            filepath = None
            uptodate = None

            for location in search_path:
                try:
                    if isinstance(location, string_types):
                        filepath = os.path.join(location, filename)
                        try:
                            fileobj = open(filepath, 'rb')
                            mtime = os.path.getmtime(filepath)
                        except IOError:
                            continue

                        def uptodate(filepath=filepath, mtime=mtime):
                            return mtime == os.path.getmtime(filepath)
                    else:
                        fileobj, filepath, uptodate = location(filename)
                except TemplateNotFound:
                    continue
                else:
                    break
            else:
                raise TemplateNotFound(filename, search_path)

            if isabs:
                filename = filepath

            try:
                template = cls(
                    fileobj,
                    basedir=os.path.dirname(filepath),
                    filename=filename,
                    loader=self,
                    encoding=encoding or self.default_encoding,
                    lookup=self.variable_lookup,
                    allow_exec=self.allow_exec
                )
            finally:
                fileobj.close()

            if self.callback is not None:
                self.callback(template)

            self._cache[cachekey] = template
            self._uptodate[cachekey] = uptodate
            return template
        finally:
            self._lock.release()
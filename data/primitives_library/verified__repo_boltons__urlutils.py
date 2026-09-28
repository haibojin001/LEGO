import re
import socket
import string
from unicodedata import normalize
from collections.abc import MutableMapping

_UNRESERVED_CHARS = frozenset(
    '~-._0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
)
_GEN_DELIMS = frozenset(':/?#[]@')
_SUB_DELIMS = frozenset("!$&'()*+,;=")
_ALL_DELIMS = _GEN_DELIMS | _SUB_DELIMS

_USERINFO_SAFE = _UNRESERVED_CHARS | _SUB_DELIMS
_USERINFO_DELIMS = _ALL_DELIMS - _USERINFO_SAFE
_PATH_SAFE = _UNRESERVED_CHARS | _SUB_DELIMS | set(':@')
_PATH_DELIMS = _ALL_DELIMS - _PATH_SAFE
_FRAGMENT_SAFE = _UNRESERVED_CHARS | _PATH_SAFE | set('/?')
_FRAGMENT_DELIMS = _ALL_DELIMS - _FRAGMENT_SAFE
_QUERY_SAFE = _UNRESERVED_CHARS | _FRAGMENT_SAFE - set('&=+')
_QUERY_DELIMS = _ALL_DELIMS - _QUERY_SAFE

SCHEME_PORT_MAP = {
    'acap': 674, 'afp': 548, 'dict': 2628, 'dns': 53, 'file': None,
    'ftp': 21, 'git': 9418, 'gopher': 70, 'http': 80, 'https': 443,
    'imap': 143, 'ipp': 631, 'ipps': 631, 'irc': 194, 'ircs': 6697,
    'ldap': 389, 'ldaps': 636, 'mms': 1755, 'msrp': 2855,
    'msrps': None, 'mtqp': 1038, 'nfs': 111, 'nntp': 119,
    'nntps': 563, 'pop': 110, 'prospero': 1525, 'redis': 6379,
    'rsync': 873, 'rtsp': 554, 'rtsps': 322, 'rtspu': 5005,
    'sftp': 22, 'smb': 445, 'snmp': 161, 'ssh': 22, 'steam': None,
    'svn': 3690, 'telnet': 23, 'ventrilo': 3784, 'vnc': 5900,
    'wais': 210, 'ws': 80, 'wss': 443, 'xmpp': None,
}

NO_NETLOC_SCHEMES = {
    'urn', 'about', 'bitcoin', 'blob', 'data', 'geo', 'magnet',
    'mailto', 'news', 'pkcs11', 'sip', 'sips', 'tel',
}

DEFAULT_ENCODING = 'utf8'

_URL_RE = re.compile(
    r'^((?P<scheme>[^:/?#]+):)?'
    r'((?P<_netloc_sep>//)(?P<authority>[^/?#]*))?'
    r'(?P<path>[^?#]*)'
    r'(\?(?P<query>[^#]*))?'
    r'(#(?P<fragment>.*))?'
)

_FIND_ALL_URL_RE = re.compile(
    r"""\b((?:([\w-]+):(/{1,3})|www[.])(?:(?:(?:[^\s&()<>]|&amp;|&quot;)*(?:[^!"#$%'()*+,.:;<=>?@\[\]^`{|}~\s]))|(?:\((?:[^\s&()]|&amp;|&quot;)*\)))+)"""
)

_HEX = frozenset(string.hexdigits)
_HEX_CHAR_MAP = {
    (a + b).encode('ascii'): chr(int(a + b, 16)).encode('latin1')
    for a in string.hexdigits for b in string.hexdigits
}
_ASCII_RE = re.compile('([\x00-\x7f]+)')


class URLParseError(ValueError):
    pass


def to_unicode(obj):
    try:
        return str(obj)
    except UnicodeDecodeError:
        return str(obj, encoding=DEFAULT_ENCODING)


def _make_quote_map(safe_chars):
    ret = {}
    for i in range(256):
        c = chr(i)
        ret[c] = ret[i] = c if c in safe_chars else '%%%02X' % i
    return ret


_USERINFO_PART_QUOTE_MAP = _make_quote_map(_USERINFO_SAFE)
_PATH_PART_QUOTE_MAP = _make_quote_map(_PATH_SAFE)
_QUERY_PART_QUOTE_MAP = _make_quote_map(_QUERY_SAFE)
_FRAGMENT_PART_QUOTE_MAP = _make_quote_map(_FRAGMENT_SAFE)


def quote(string, safe='/', full_quote=True):
    if string is None:
        return ''
    if isinstance(string, bytes):
        raw = string
    else:
        raw = normalize('NFC', to_unicode(string)).encode(DEFAULT_ENCODING)
    safe_set = set(safe) | _UNRESERVED_CHARS
    if not full_quote:
        safe_set.add('%')
    ret = []
    for b in raw:
        c = chr(b)
        if c in safe_set:
            ret.append(c)
        else:
            ret.append('%%%02X' % b)
    return ''.join(ret)


def quote_path_part(string, full_quote=True):
    if string is None:
        return ''
    if not full_quote:
        return quote(string, ''.join(_PATH_SAFE | {'%'}), False)
    return quote(string, ''.join(_PATH_SAFE), True)


def quote_query_part(string, full_quote=True):
    if string is None:
        return ''
    if not full_quote:
        return quote(string, ''.join(_QUERY_SAFE | {'%'}), False)
    return quote(string, ''.join(_QUERY_SAFE), True)


def quote_fragment_part(string, full_quote=True):
    if string is None:
        return ''
    if not full_quote:
        return quote(string, ''.join(_FRAGMENT_SAFE | {'%'}), False)
    return quote(string, ''.join(_FRAGMENT_SAFE), True)


def quote_userinfo_part(string, full_quote=True):
    if string is None:
        return ''
    if not full_quote:
        return quote(string, ''.join(_USERINFO_SAFE | {'%'}), False)
    return quote(string, ''.join(_USERINFO_SAFE), True)


def unquote_to_bytes(string):
    if string is None:
        return b''
    if isinstance(string, bytes):
        raw = string
    else:
        raw = to_unicode(string).encode('utf8')
    out = bytearray()
    i = 0
    length = len(raw)
    while i < length:
        if raw[i:i + 1] == b'%' and i + 2 < length:
            pair = raw[i + 1:i + 3]
            value = _HEX_CHAR_MAP.get(pair)
            if value is not None:
                out.extend(value)
                i += 3
                continue
        out.extend(raw[i:i + 1])
        i += 1
    return bytes(out)


def unquote(string, encoding='utf-8', errors='replace'):
    if string is None:
        return None
    return unquote_to_bytes(string).decode(encoding, errors)


def parse_qsl(qs, keep_blank_values=True, strict_parsing=False):
    if qs is None:
        return []
    qs = to_unicode(qs)
    pairs = []
    for field in re.split('[&;]', qs):
        if not field:
            continue
        if '=' in field:
            name, value = field.split('=', 1)
        else:
            if strict_parsing:
                raise ValueError('bad query field: %r' % field)
            name, value = field, ''
        if value or keep_blank_values:
            pairs.append((unquote(name.replace('+', ' ')),
                          unquote(value.replace('+', ' '))))
    return pairs


class QueryParamDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._pairs = []
        if args:
            if len(args) > 1:
                raise TypeError('expected at most 1 argument')
            source = args[0]
            if isinstance(source, str):
                self._pairs.extend(parse_qsl(source))
            elif hasattr(source, 'items'):
                for key, value in source.items():
                    if isinstance(value, (list, tuple)):
                        for item in value:
                            self.add(key, item)
                    else:
                        self.add(key, value)
            else:
                for key, value in source:
                    self.add(key, value)
        for key, value in kwargs.items():
            self.add(key, value)

    @classmethod
    def from_text(cls, query_string):
        return cls(query_string)

    def to_text(self, full_quote=True):
        return '&'.join(
            '%s=%s' % (quote_query_part(k, full_quote),
                       quote_query_part(v, full_quote))
            for k, v in self._pairs
        )

    def add(self, key, value):
        self._pairs.append((to_unicode(key), to_unicode(value)))

    def addlist(self, key, values):
        for value in values:
            self.add(key, value)

    def getlist(self, key, default=None):
        values = [v for k, v in self._pairs if k == key]
        return values if values else ([] if default is None else default)

    def setlist(self, key, values):
        self.pop(key, None)
        self.addlist(key, values)

    def items(self, multi=False):
        if multi:
            return list(self._pairs)
        seen = set()
        ret = []
        for key, value in reversed(self._pairs):
            if key not in seen:
                seen.add(key)
                ret.append((key, value))
        ret.reverse()
        return ret

    def iteritems(self, multi=False):
        return iter(self.items(multi=multi))

    def iterkeys(self, multi=False):
        if multi:
            return (k for k, v in self._pairs)
        return iter(self.keys())

    def itervalues(self, multi=False):
        if multi:
            return (v for k, v in self._pairs)
        return iter(self.values())

    def keys(self):
        return [key for key, value in self.items()]

    def values(self):
        return [value for key, value in self.items()]

    def copy(self):
        return self.__class__(self._pairs)

    def __getitem__(self, key):
        for cur_key, value in reversed(self._pairs):
            if cur_key == key:
                return value
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.pop(key, None)
        self.add(key, value)

    def __delitem__(self, key):
        old_len = len(self._pairs)
        self._pairs[:] = [(k, v) for k, v in self._pairs if k != key]
        if len(self._pairs) == old_len:
            raise KeyError(key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._pairs)

    def __eq__(self, other):
        if isinstance(other, QueryParamDict):
            return self._pairs == other._pairs
        return dict(self.items()) == other


def register_scheme(text, default_port=None, uses_netloc=True):
    scheme = to_unicode(text).lower()
    if uses_netloc:
        SCHEME_PORT_MAP[scheme] = default_port
        NO_NETLOC_SCHEMES.discard(scheme)
    else:
        NO_NETLOC_SCHEMES.add(scheme)
        SCHEME_PORT_MAP.pop(scheme, None)


def parse_host(host):
    host = to_unicode(host)
    if not host:
        return host
    if host.startswith('[') and host.endswith(']'):
        host = host[1:-1]
    try:
        socket.inet_pton(socket.AF_INET, host)
        return host
    except (OSError, AttributeError):
        pass
    try:
        socket.inet_pton(socket.AF_INET6, host.split('%', 1)[0])
        return host
    except (OSError, AttributeError):
        pass
    try:
        return host.encode('ascii').decode('idna')
    except UnicodeError:
        return host


def resolve_path(path_parts):
    parts = list(path_parts)
    if not parts:
        return ()
    absolute = parts[0] == ''
    output = []
    for part in parts:
        if part == '.':
            continue
        if part == '..':
            if output and output[-1] not in ('', '..'):
                output.pop()
            elif not absolute:
                output.append(part)
        else:
            output.append(part)
    if absolute and (not output or output[0] != ''):
        output.insert(0, '')
    if parts and parts[-1] in ('.', '..') and output and output[-1] != '':
        output.append('')
    return tuple(output)


class URL:
    def __init__(self, url=''):
        if isinstance(url, URL):
            self.scheme = url.scheme
            self.username = url.username
            self.password = url.password
            self.host = url.host
            self.port = url.port
            self.path_parts = url.path_parts
            self.fragment = url.fragment
            self._uses_netloc = url._uses_netloc
            self._query_params = url.query_params.copy()
            return
        parsed = self.from_text(url)
        self.__dict__.update(parsed.__dict__)

    @classmethod
    def from_text(cls, url):
        text = to_unicode(url)
        match = _URL_RE.match(text)
        if not match:
            raise URLParseError('could not parse URL: %r' % text)
        groups = match.groupdict()
        obj = cls.__new__(cls)
        obj.scheme = (groups.get('scheme') or '').lower()
        obj.username = None
        obj.password = None
        obj.host = None
        obj.port = None
        obj.fragment = unquote(groups.get('fragment') or '')
        obj._query_params = QueryParamDict(groups.get('query') or '')
        obj._uses_netloc = bool(groups.get('_netloc_sep'))
        authority = groups.get('authority')
        if authority is not None:
            obj._parse_authority(authority)
        elif obj.scheme and obj.scheme not in NO_NETLOC_SCHEMES:
            obj._uses_netloc = True
        path = groups.get('path') or ''
        obj.path_parts = tuple(unquote(part) for part in path.split('/')) if path else ()
        return obj

    @classmethod
    def from_parts(cls, scheme=None, host=None, port=None, path=(),
                   query=None, fragment='', username=None, password=None):
        obj = cls.__new__(cls)
        obj.scheme = (scheme or '').lower()
        obj.host = parse_host(host) if host is not None else None
        obj.port = int(port) if port is not None else None
        if obj.port is not None and not 0 <= obj.port <= 65535:
            raise URLParseError('port out of range: %r' % obj.port)
        obj.username = None if username is None else to_unicode(username)
        obj.password = None if password is None else to_unicode(password)
        if isinstance(path, str):
            obj.path_parts = tuple(path.split('/')) if path else ()
        else:
            obj.path_parts = tuple(to_unicode(p) for p in path)
        obj.fragment = '' if fragment is None else to_unicode(fragment)
        if query is None:
            obj._query_params = QueryParamDict()
        elif isinstance(query, QueryParamDict):
            obj._query_params = query.copy()
        else:
            obj._query_params = QueryParamDict(query)
        obj._uses_netloc = bool(host is not None or username is not None or
                                password is not None or
                                (obj.scheme and obj.scheme not in NO_NETLOC_SCHEMES))
        return obj

    def _parse_authority(self, authority):
        hostport = authority
        if '@' in authority:
            userinfo, hostport = authority.rsplit('@', 1)
            if ':' in userinfo:
                username, password = userinfo.split(':', 1)
                self.username = unquote(username)
                self.password = unquote(password)
            else:
                self.username = unquote(userinfo)
        if hostport.startswith('['):
            close = hostport.find(']')
            if close < 0:
                raise URLParseError('invalid IPv6 host: %r' % hostport)
            host = hostport[1:close]
            remain = hostport[close + 1:]
            try:
                socket.inet_pton(socket.AF_INET6, host.split('%', 1)[0])
            except (OSError, AttributeError):
                raise URLParseError('invalid IPv6 host: %r' % host)
            if remain:
                if not remain.startswith(':'):
                    raise URLParseError('invalid authority: %r' % authority)
                port_text = remain[1:]
                self.port = self._parse_port(port_text)
            self.host = host.lower()
            return
        if ':' in hostport:
            host, port_text = hostport.rsplit(':', 1)
            if ':' in host:
                raise URLParseError('invalid IPv6 host: %r' % hostport)
            self.host = parse_host(host).lower() if host else ''
            self.port = self._parse_port(port_text)
        else:
            self.host = parse_host(hostport).lower() if hostport else ''

    @staticmethod
    def _parse_port(port_text):
        if not port_text or not port_text.isdigit():
            raise URLParseError('invalid port: %r' % port_text)
        port = int(port_text)
        if port > 65535:
            raise URLParseError('port out of range: %r' % port)
        return port

    @property
    def uses_netloc(self):
        return self._uses_netloc

    @property
    def default_port(self):
        return SCHEME_PORT_MAP.get(self.scheme)

    @property
    def path(self):
        return '/'.join(self.path_parts)

    @path.setter
    def path(self, value):
        self.path_parts = tuple(to_unicode(value).split('/')) if value else ()

    @property
    def query_params(self):
        return self._query_params

    @query_params.setter
    def query_params(self, value):
        self._query_params = value if isinstance(value, QueryParamDict) else QueryParamDict(value)

    @property
    def query(self):
        return self.query_params.to_text()

    @query.setter
    def query(self, value):
        self._query_params = QueryParamDict(value)

    @property
    def authority(self):
        return self.get_authority()

    def get_authority(self, full_quote=True, with_userinfo=True):
        parts = []
        if with_userinfo and self.username is not None:
            user = quote_userinfo_part(self.username, full_quote)
            if self.password is not None:
                user += ':' + quote_userinfo_part(self.password, full_quote)
            parts.append(user + '@')
        host = self.host or ''
        if ':' in host and not host.startswith('['):
            host = '[' + host + ']'
        elif host:
            try:
                host = host.encode('idna').decode('ascii')
            except UnicodeError:
                pass
        parts.append(host)
        if self.port is not None:
            parts.append(':' + str(self.port))
        return ''.join(parts)

    def to_text(self, full_quote=True):
        ret = []
        if self.scheme:
            ret.append(self.scheme)
            ret.append(':')
        if self._uses_netloc:
            ret.append('//')
            ret.append(self.get_authority(full_quote=full_quote))
        if self.path_parts:
            ret.append('/'.join(quote_path_part(part, full_quote)
                                for part in self.path_parts))
        query = self.query_params.to_text(full_quote=full_quote)
        if query:
            ret.append('?')
            ret.append(query)
        if self.fragment:
            ret.append('#')
            ret.append(quote_fragment_part(self.fragment, full_quote))
        return ''.join(ret)

    def normalize(self, with_case=True):
        ret = self.copy()
        if with_case:
            ret.scheme = ret.scheme.lower()
            if ret.host is not None:
                ret.host = ret.host.lower()
        ret.path_parts = resolve_path(ret.path_parts)
        if ret.port is not None and ret.port == ret.default_port:
            ret.port = None
        return ret

    def navigate(self, other):
        if not isinstance(other, URL):
            other = URL(other)
        if other.scheme:
            return other.copy()
        ret = self.copy()
        if other._uses_netloc:
            ret.username = other.username
            ret.password = other.password
            ret.host = other.host
            ret.port = other.port
            ret.path_parts = other.path_parts
            ret.query_params = other.query_params.copy()
            return ret
        if other.path_parts:
            if other.path_parts[0] == '':
                ret.path_parts = resolve_path(other.path_parts)
            else:
                base = list(ret.path_parts)
                if base:
                    base.pop()
                ret.path_parts = resolve_path(tuple(base) + other.path_parts)
            ret.query_params = other.query_params.copy()
        elif other.query_params._pairs:
            ret.query_params = other.query_params.copy()
        ret.fragment = other.fragment
        return ret

    def copy(self):
        return self.__class__(self)

    def get_query_param(self, name, default=None):
        return self.query_params.get(name, default)

    def add_query_param(self, name, value):
        ret = self.copy()
        ret.query_params.add(name, value)
        return ret

    def set_query_param(self, name, value):
        ret = self.copy()
        ret.query_params[name] = value
        return ret

    def remove_query_param(self, name):
        ret = self.copy()
        ret.query_params.pop(name, None)
        return ret

    def add_path(self, *parts):
        ret = self.copy()
        new_parts = list(ret.path_parts)
        for part in parts:
            if isinstance(part, (tuple, list)):
                new_parts.extend(to_unicode(p) for p in part)
            else:
                new_parts.extend(to_unicode(part).split('/'))
        ret.path_parts = tuple(new_parts)
        return ret

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.to_text())

    def __eq__(self, other):
        if not isinstance(other, URL):
            try:
                other = URL(other)
            except Exception:
                return False
        return self.to_text() == other.to_text()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.to_text())


def find_all_links(text, with_text=False, default_scheme='https', schemes=()):
    text = to_unicode(text)
    prev_end = 0
    ret = []

    def add_text(value):
        if ret and isinstance(ret[-1], str):
            ret[-1] += value
        else:
            ret.append(value)

    for match in _FIND_ALL_URL_RE.finditer(text):
        start, end = match.start(1), match.end(1)
        if prev_end < start and with_text:
            add_text(text[prev_end:start])
        prev_end = end
        try:
            url_text = match.group(0)
            url = URL(url_text)
            if not url.scheme:
                if default_scheme:
                    url = URL(default_scheme + '://' + url_text)
                else:
                    if with_text:
                        add_text(text[start:end])
                    continue
            if schemes and url.scheme not in schemes:
                if with_text:
                    add_text(text[start:end])
            else:
                ret.append(url)
        except URLParseError:
            if with_text:
                add_text(text[start:end])

    if with_text:
        tail = text[prev_end:]
        if tail:
            add_text(tail)
    return ret
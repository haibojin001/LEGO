# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg528::inspect.getsourcefile+inspect.getsourcelines+operator.attrgetter
# name: inspect_operator_primitive
# summary: Uses inspect.getsourcefile, inspect.getsourcelines, operator.attrgetter across 2 repos
# anchor_symbols: ['inspect.getsourcefile', 'inspect.getsourcelines', 'operator.attrgetter']
# observed in 2 repos: ['ZhiningLiu1998__imbalanced-ensemble', 'feature-engine__feature_engine']...

# --- from feature-engine__feature_engine::docs/sphinxext/github_link.py::_linkcode_resolve ---
def _linkcode_resolve(domain, info, package, url_fmt, revision):
    """Determine a link to online source for a class/method/function

    This is called by sphinx.ext.linkcode

    An example with a long-untouched module that everyone has
    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='http://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'http://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """

    if revision is None:
        return
    if domain not in ('py', 'pyx'):
        return
    if not info.get('module') or not info.get('fullname'):
        return

    class_name = info['fullname'].split('.')[0]
    if type(class_name) != str:
        # Python 2 only
        class_name = class_name.encode('utf-8')
    module = __import__(info['module'], fromlist=[class_name])
    obj = attrgetter(info['fullname'])(module)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return

    fn = os.path.relpath(fn,
                         start=os.path.dirname(__import__(package).__file__))
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ''
    return url_fmt.format(revision=revision, package=package,
                          path=fn, lineno=lineno)

# --- from ZhiningLiu1998__imbalanced-ensemble::docs/source/sphinxext/github_link.py::_linkcode_resolve ---
def _linkcode_resolve(domain, info, package, url_fmt, revision):
    """Determine a link to online source for a class/method/function

    This is called by sphinx.ext.linkcode

    An example with a long-untouched module that everyone has
    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='http://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'http://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """

    if revision is None:
        return
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    class_name = info["fullname"].split(".")[0]
    if type(class_name) != str:
        # Python 2 only
        class_name = class_name.encode("utf-8")
    module = __import__(info["module"], fromlist=[class_name])
    obj = attrgetter(info["fullname"])(module)

    # try:
    #     fn = inspect.getsourcefile(obj)
    # except Exception:
    #     fn = None
    # if not fn:
    #     try:
    #         fn = inspect.getsourcefile(sys.modules[obj.__module__])
    #     except Exception:
    #         fn = None
    try:
        fn = inspect.getsourcefile(sys.modules[obj.__module__])
    except Exception:
        fn = None
    if not fn:
        return

    fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))
    try:
        src_code_lines, lineno = inspect.getsourcelines(obj)
        i = 0
        for l in src_code_lines:
            if 'def' in l or 'class' in l:
                break
            i += 1
        lineno += i
    except Exception:
        lineno = ""
    return url_fmt.format(revision=revision, package=package, path=fn, lineno=lineno)

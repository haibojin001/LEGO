import re
import os
import sys
import json
import time
import random
import socket
import struct
import getpass
import datetime
import platform

ECO_VERSION = '1.1.0'


try:
    getrandbits = random.SystemRandom().getrandbits
    HAVE_URANDOM = True
except Exception:
    HAVE_URANDOM = False
    getrandbits = random.getrandbits


INSTANCE_ID = hex(getrandbits(128))[2:-1].lower()

IS_64BIT = struct.calcsize('P') > 4
HAVE_UCS4 = getattr(sys, 'maxunicode', 0) > 65536
HAVE_READLINE = True

try:
    import readline
except Exception:
    HAVE_READLINE = False

try:
    import sqlite3
    SQLITE_VERSION = sqlite3.sqlite_version
except Exception:
    SQLITE_VERSION = ''

try:
    import ssl
    try:
        OPENSSL_VERSION = ssl.OPENSSL_VERSION
    except AttributeError:
        OPENSSL_VERSION = 'OpenSSL >0.8.0'
except Exception:
    OPENSSL_VERSION = ''

try:
    import tkinter
    TKINTER_VERSION = str(tkinter.TkVersion)
except Exception:
    TKINTER_VERSION = ''

try:
    import zlib
    ZLIB_VERSION = zlib.ZLIB_VERSION
except Exception:
    ZLIB_VERSION = ''

try:
    from xml.parsers import expat
    EXPAT_VERSION = expat.EXPAT_VERSION
except Exception:
    EXPAT_VERSION = ''

try:
    from multiprocessing import cpu_count
    CPU_COUNT = cpu_count()
except Exception:
    CPU_COUNT = 0

try:
    import threading
    HAVE_THREADING = True
except Exception:
    HAVE_THREADING = False

try:
    HAVE_IPV6 = socket.has_ipv6
except Exception:
    HAVE_IPV6 = False

try:
    from resource import getrlimit, RLIMIT_NOFILE
    RLIMIT_FDS_SOFT, RLIMIT_FDS_HARD = getrlimit(RLIMIT_NOFILE)
except Exception:
    RLIMIT_FDS_SOFT, RLIMIT_FDS_HARD = 0, 0

START_TIME_INFO = {
    'time_utc': str(datetime.datetime.now(datetime.timezone.utc)),
    'time_utc_offset': -time.timezone / 3600.0,
}


def _escape_shell_args(args):
    return ' '.join(
        re.sub(r'([^\w@%+=:,./-])', r'\\\1', arg)
        for arg in args
    )


def _get_umask():
    current_umask = os.umask(0)
    os.umask(current_umask)
    return '%03o' % current_umask


def _get_linux_dist():
    if not sys.platform.startswith('linux'):
        return '', ''

    try:
        with open('/etc/os-release', encoding='utf-8') as os_release_file:
            os_release = os_release_file.read()
    except Exception:
        return '', ''

    values = {}
    for line in os_release.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        value = value.strip()
        if (len(value) >= 2
                and value[0] == value[-1]
                and value[0] in ("'", '"')):
            value = value[1:-1]
            value = re.sub(r'\\(["\\$`])', r'\1', value)
        values[key] = value

    return values.get('NAME', ''), values.get('VERSION_ID', '')


def get_python_info():
    ret = {}
    ret['argv'] = _escape_shell_args(sys.argv)
    ret['bin'] = sys.executable
    ret['version'] = ' '.join(sys.version.split())
    ret['compiler'] = platform.python_compiler()
    ret['build_date'] = platform.python_build()[1]
    ret['version_info'] = list(sys.version_info)
    ret['features'] = {
        'openssl': OPENSSL_VERSION,
        'expat': EXPAT_VERSION,
        'sqlite': SQLITE_VERSION,
        'tkinter': TKINTER_VERSION,
        'zlib': ZLIB_VERSION,
        'unicode_wide': HAVE_UCS4,
        'readline': HAVE_READLINE,
        '64bit': IS_64BIT,
        'ipv6': HAVE_IPV6,
        'threading': HAVE_THREADING,
        'urandom': HAVE_URANDOM,
    }
    return ret


def get_profile(**kwargs):
    uname = platform.uname()
    ret = {
        '_eco_version': ECO_VERSION,
        'guid': INSTANCE_ID,
        'uname': dict(zip(
            ('system', 'node', 'release', 'version', 'machine', 'processor'),
            uname,
        )),
        'hostname': socket.gethostname(),
        'hostfqdn': socket.getfqdn(),
        'username': getpass.getuser(),
        'cpu_count': CPU_COUNT,
        'cwd': os.getcwd(),
        'umask': _get_umask(),
        'python': get_python_info(),
        'fs_encoding': sys.getfilesystemencoding(),
        'ulimit_soft': RLIMIT_FDS_SOFT,
        'ulimit_hard': RLIMIT_FDS_HARD,
    }

    if sys.platform.startswith('linux'):
        linux_dist_name, linux_dist_version = _get_linux_dist()
        ret['linux_dist_name'] = linux_dist_name
        ret['linux_dist_version'] = linux_dist_version

    ret.update(START_TIME_INFO)
    ret.update(kwargs)
    return ret


if __name__ == '__main__':
    print(json.dumps(get_profile(), indent=2, sort_keys=True))
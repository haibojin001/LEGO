from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ssl as _ssl_module
    from ssl import (
        SSLError as _SSLErrorType,
        SSLEOFError as _SSLEOFErrorType,
        SSLWantReadError as _SSLWantReadErrorType,
        SSLWantWriteError as _SSLWantWriteErrorType,
    )
else:
    _ssl_module = None
    _SSLErrorType = None
    _SSLEOFErrorType = None
    _SSLWantReadErrorType = None
    _SSLWantWriteErrorType = None

__all__ = (
    "HAVE_SSL",
    "ssl",
    "SSLError",
    "SSLEOFError",
    "SSLWantReadError",
    "SSLWantWriteError",
)

try:
    import ssl
    from ssl import SSLError, SSLEOFError, SSLWantReadError, SSLWantWriteError
except ImportError:
    HAVE_SSL = False
    ssl = None

    class SSLError(Exception):
        pass

    class SSLEOFError(Exception):
        pass

    class SSLWantReadError(Exception):
        pass

    class SSLWantWriteError(Exception):
        pass
else:
    HAVE_SSL = True
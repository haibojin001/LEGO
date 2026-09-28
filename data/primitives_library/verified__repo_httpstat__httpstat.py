from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import NoReturn, overload


__version__ = "2.0.0"


class Env:
    prefix = "HTTPSTAT"
    _instances: list["Env"] = []

    def __init__(self, key: str):
        self.key = key.format(prefix=self.prefix)
        Env._instances.append(self)

    @overload
    def get(self, default: str) -> str: ...

    @overload
    def get(self, default: None = None) -> str | None: ...

    def get(self, default: str | None = None) -> str | None:
        return os.environ.get(self.key, default)


ENV_SHOW_BODY = Env("{prefix}_SHOW_BODY")
ENV_SHOW_IP = Env("{prefix}_SHOW_IP")
ENV_SHOW_SPEED = Env("{prefix}_SHOW_SPEED")
ENV_SAVE_BODY = Env("{prefix}_SAVE_BODY")
ENV_CURL_BIN = Env("{prefix}_CURL_BIN")
ENV_METRICS_ONLY = Env("{prefix}_METRICS_ONLY")
ENV_DEBUG = Env("{prefix}_DEBUG")


curl_format = """{
"time_namelookup": %{time_namelookup},
"time_connect": %{time_connect},
"time_appconnect": %{time_appconnect},
"time_pretransfer": %{time_pretransfer},
"time_redirect": %{time_redirect},
"time_starttransfer": %{time_starttransfer},
"time_total": %{time_total},
"speed_download": %{speed_download},
"speed_upload": %{speed_upload},
"remote_ip": "%{remote_ip}",
"remote_port": "%{remote_port}",
"local_ip": "%{local_ip}",
"local_port": "%{local_port}"
}"""

https_template = """
  DNS Lookup   TCP Connection   TLS Handshake   Server Processing   Content Transfer
[   {a0000}  |     {a0001}    |    {a0002}    |      {a0003}      |      {a0004}     ]
             |                |               |                   |                  |
    namelookup:{b0000}        |               |                   |                  |
                        connect:{b0001}       |                   |                  |
                                    pretransfer:{b0002}           |                  |
                                                      starttransfer:{b0003}          |
                                                                                 total:{b0004}
"""[1:]

http_template = """
  DNS Lookup   TCP Connection   Server Processing   Content Transfer
[   {a0000}  |     {a0001}    |      {a0003}      |      {a0004}     ]
             |                |                   |                  |
    namelookup:{b0000}        |                   |                  |
                        connect:{b0001}           |                  |
                                      starttransfer:{b0003}          |
                                                                 total:{b0004}
"""[1:]


ISATTY = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def make_color(code):
    def color_func(s):
        if not ISATTY:
            return s
        return f"\x1b[{code}m{s}\x1b[0m"

    return color_func


red = make_color(31)
green = make_color(32)
yellow = make_color(33)
blue = make_color(34)
magenta = make_color(35)
cyan = make_color(36)

bold = make_color(1)
underline = make_color(4)

grayscale = {(i - 232): make_color(f"38;5;{i}") for i in range(232, 256)}

_TRUTHY = frozenset(("1", "true", "yes", "on"))
_FALSY = frozenset(("0", "false", "no", "off"))


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def pop_arg(args: list[str], flag: str, has_value: bool = True) -> str | bool | None:
    if flag not in args:
        return None
    index = args.index(flag)
    if has_value:
        if index + 1 >= len(args):
            return None
        args.pop(index)
        return args.pop(index)
    args.pop(index)
    return True


SLO_KEY_MAP = {
    "total": "time_total",
    "connect": "time_connect",
    "ttfb": "time_starttransfer",
    "dns": "time_namelookup",
    "tls": "time_pretransfer",
}


def parse_slo(spec: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            print("Error: empty SLO spec")
            sys.exit(1)
        if "=" not in item:
            print(f'Error: invalid SLO spec "{item}", expected key=value')
            sys.exit(1)
        name, _, value = item.partition("=")
        name = name.strip()
        value = value.strip()
        if name not in SLO_KEY_MAP:
            allowed = ", ".join(SLO_KEY_MAP)
            print(f'Error: unknown SLO key "{name}", valid keys: {allowed}')
            sys.exit(1)
        try:
            milliseconds = int(value)
        except ValueError:
            print(
                f'Error: SLO value for "{name}" must be a positive integer, '
                f'got "{value}"'
            )
            sys.exit(1)
        if milliseconds <= 0:
            print(f'Error: SLO value for "{name}" must be positive, got {milliseconds}')
            sys.exit(1)
        parsed[name] = milliseconds
    return parsed


def check_slo(slo: dict[str, int], timings: dict) -> tuple[bool, list[dict]]:
    failures: list[dict] = []
    for name, maximum in slo.items():
        actual = timings[SLO_KEY_MAP[name]]
        if actual > maximum:
            failures.append(
                {
                    "key": name,
                    "threshold_ms": maximum,
                    "actual_ms": actual,
                }
            )
    return not failures, failures


def build_json_result(
    url: str,
    d: dict,
    headers_text: str,
    slo_result: tuple[bool, list[dict]] | None,
    exit_code: int,
) -> dict:
    first_line = headers_text.split("\n")[0].strip().rstrip("\r")
    parts = first_line.split(None, 2)
    try:
        status_code = int(parts[1]) if len(parts) >= 2 else 0
    except (ValueError, IndexError):
        status_code = 0

    headers: dict[str, str] = {}
    for line in headers_text.split("\n")[1:]:
        line = line.strip().rstrip("\r")
        if not line:
            continue
        separator = line.find(":")
        if separator >= 0:
            headers[line[:separator].strip()] = line[separator + 1 :].strip()

    result = {
        "schema_version": 1,
        "url": url,
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "response": {
            "status_line": first_line,
            "status_code": status_code,
            "remote_ip": d.get("remote_ip", ""),
            "remote_port": d.get("remote_port", ""),
            "headers": headers,
        },
        "timings_ms": {
            "dns": d["range_dns"],
            "connect": d["range_connection"],
            "tls": d["range_ssl"],
            "server": d["range_server"],
            "transfer": d["range_transfer"],
            "total": d["time_total"],
            "namelookup": d["time_namelookup"],
            "initial_connect": d["time_connect"],
            "pretransfer": d["time_pretransfer"],
            "starttransfer": d["time_starttransfer"],
        },
        "speed": {
            "download_kbs": round(d.get("speed_download", 0) / 1024, 1),
            "upload_kbs": round(d.get("speed_upload", 0) / 1024, 1),
        },
        "slo": None,
    }
    if slo_result is not None:
        result["slo"] = {"pass": slo_result[0], "violations": slo_result[1]}
    return result


def _exit(message, code: int = 0) -> NoReturn:
    if message is not None:
        print(message)
    sys.exit(code)


def print_help():
    print(
        """Usage: httpstat URL [CURL_OPTIONS]
       httpstat -h | --help
       httpstat --version

Arguments:
  URL     url to request, could be with or without `http(s)://` prefix

Options:
  CURL_OPTIONS  any curl supported options, except for -w -D -o -S -s,
                which are already used internally.
  -h --help     show this screen.
  --version     show version.
  -f --format   output format: pretty, json, jsonl. Default is `pretty`.
  --slo         SLO thresholds in milliseconds, e.g. total=500,connect=100.

Environment:
  HTTPSTAT_SHOW_BODY      show response body.
  HTTPSTAT_SHOW_IP        show local and remote IP addresses.
  HTTPSTAT_SHOW_SPEED     show upload and download speeds.
  HTTPSTAT_SAVE_BODY      save response body to this file.
  HTTPSTAT_CURL_BIN       curl executable to use.
  HTTPSTAT_METRICS_ONLY   suppress response headers and body.
  HTTPSTAT_DEBUG          print curl command and diagnostic information.
"""
    )


def _environment_enabled(env: Env) -> bool:
    value = env.get()
    if value is None:
        return False
    try:
        return parse_bool(value)
    except ValueError:
        return bool(value)


def _format_ms(value) -> str:
    try:
        return f"{float(value):.0f} ms"
    except (TypeError, ValueError):
        return "0 ms"


def _range_color(value: float):
    if value < 10:
        return grayscale[15]
    if value < 100:
        return cyan
    if value < 500:
        return yellow
    return red


def _colored_range(value) -> str:
    text = _format_ms(value)
    try:
        return _range_color(float(value))(text)
    except (TypeError, ValueError):
        return text


def _metric_data(raw: dict) -> dict:
    data = dict(raw)
    for key, value in tuple(data.items()):
        if key.startswith("time_"):
            try:
                data[key] = float(value) * 1000
            except (TypeError, ValueError):
                data[key] = 0
    for key in ("speed_download", "speed_upload"):
        try:
            data[key] = float(data.get(key, 0))
        except (TypeError, ValueError):
            data[key] = 0

    lookup = data.get("time_namelookup", 0)
    connect = data.get("time_connect", 0)
    appconnect = data.get("time_appconnect", 0)
    pretransfer = data.get("time_pretransfer", 0)
    starttransfer = data.get("time_starttransfer", 0)
    total = data.get("time_total", 0)

    data["range_dns"] = lookup
    data["range_connection"] = max(0, connect - lookup)
    data["range_ssl"] = max(0, appconnect - connect)
    data["range_server"] = max(0, starttransfer - pretransfer)
    data["range_transfer"] = max(0, total - starttransfer)
    return data


def print_response_headers(headers: str):
    headers = headers.rstrip()
    if headers:
        print(bold(headers))


def print_body(body):
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if body:
        print(body.rstrip())


def print_stats(d: dict):
    values = {
        "a0000": _colored_range(d.get("range_dns", 0)),
        "a0001": _colored_range(d.get("range_connection", 0)),
        "a0002": _colored_range(d.get("range_ssl", 0)),
        "a0003": _colored_range(d.get("range_server", 0)),
        "a0004": _colored_range(d.get("range_transfer", 0)),
        "b0000": _format_ms(d.get("time_namelookup", 0)),
        "b0001": _format_ms(d.get("time_connect", 0)),
        "b0002": _format_ms(d.get("time_pretransfer", 0)),
        "b0003": _format_ms(d.get("time_starttransfer", 0)),
        "b0004": _format_ms(d.get("time_total", 0)),
    }
    encrypted = d.get("time_appconnect", 0) > 0
    print((https_template if encrypted else http_template).format(**values), end="")


def _read_text(path: str) -> str:
    try:
        with open(path, "rb") as stream:
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _extract_options(args: list[str]) -> tuple[str, str | None, str | None, list[str]]:
    output_format = "pretty"
    slo_spec = None

    value = pop_arg(args, "--format")
    if value is None:
        value = pop_arg(args, "-f")
    if value is not None:
        if not isinstance(value, str):
            _exit("Error: format requires a value", 1)
        output_format = value.lower()

    value = pop_arg(args, "--slo")
    if value is not None:
        if not isinstance(value, str):
            _exit("Error: --slo requires a value", 1)
        slo_spec = value

    if output_format not in ("pretty", "json", "jsonl"):
        _exit(f'Error: unsupported format "{output_format}"', 1)

    if not args:
        _exit(None, 1)

    url = args.pop(0)
    return url, output_format, slo_spec, args


def main():
    arguments = list(sys.argv[1:])

    if not arguments:
        print_help()
        return

    if arguments[0] in ("-h", "--help"):
        print_help()
        return

    if arguments[0] == "--version":
        print(__version__)
        return

    url, output_format, slo_spec, curl_args = _extract_options(arguments)

    if "://" not in url:
        url = "http://" + url

    slo = parse_slo(slo_spec) if slo_spec is not None else None
    curl_binary = ENV_CURL_BIN.get("curl")
    if shutil.which(curl_binary) is None and not os.path.isfile(curl_binary):
        _exit(f"Error: curl executable not found: {curl_binary}", 1)

    debug = _environment_enabled(ENV_DEBUG)
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    header_file = tempfile.NamedTemporaryFile(prefix="httpstat-", delete=False)
    body_file = tempfile.NamedTemporaryFile(prefix="httpstat-", delete=False)
    header_name = header_file.name
    body_name = body_file.name
    header_file.close()
    body_file.close()

    command = [
        curl_binary,
        *curl_args,
        "-w",
        curl_format,
        "-D",
        header_name,
        "-o",
        body_name,
        "-sS",
        url,
    ]

    if debug:
        logging.debug("curl command: %r", command)

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        for name in (header_name, body_name):
            try:
                os.unlink(name)
            except OSError:
                pass
        _exit(f"Error: unable to execute curl: {exc}", 1)

    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    headers = _read_text(header_name)
    body = _read_text(body_name)

    try:
        raw = json.loads(stdout)
    except (TypeError, ValueError, json.JSONDecodeError):
        raw = {}

    data = _metric_data(raw)

    save_body = ENV_SAVE_BODY.get()
    if save_body:
        try:
            shutil.copyfile(body_name, save_body)
        except OSError as exc:
            print(f"Error: unable to save response body: {exc}", file=sys.stderr)

    slo_result = check_slo(slo, data) if slo is not None and raw else None
    final_exit = completed.returncode
    if final_exit == 0 and slo_result is not None and not slo_result[0]:
        final_exit = 1

    if output_format in ("json", "jsonl"):
        result = build_json_result(url, data, headers, slo_result, final_exit)
        if output_format == "json":
            print(json.dumps(result, indent=2, sort_keys=False))
        else:
            print(json.dumps(result, separators=(",", ":"), sort_keys=False))
    else:
        if stderr:
            print(stderr.rstrip(), file=sys.stderr)

        if not _environment_enabled(ENV_METRICS_ONLY):
            if headers:
                print_response_headers(headers)
                print()
            if _environment_enabled(ENV_SHOW_BODY) and body:
                print_body(body)
                print()

        if raw:
            print_stats(data)

        if _environment_enabled(ENV_SHOW_IP):
            print()
            print(f"Remote IP: {data.get('remote_ip', '')}:{data.get('remote_port', '')}")
            print(f"Local IP:  {data.get('local_ip', '')}:{data.get('local_port', '')}")

        if _environment_enabled(ENV_SHOW_SPEED):
            print()
            print(f"Download speed: {data.get('speed_download', 0) / 1024:.1f} kB/s")
            print(f"Upload speed:   {data.get('speed_upload', 0) / 1024:.1f} kB/s")

        if slo_result is not None:
            print()
            if slo_result[0]:
                print(green("SLO: PASS"))
            else:
                print(red("SLO: FAIL"))
                for violation in slo_result[1]:
                    print(
                        f"  {violation['key']}: {violation['actual_ms']:.0f} ms "
                        f"(limit {violation['threshold_ms']} ms)"
                    )

    for name in (header_name, body_name):
        try:
            os.unlink(name)
        except OSError:
            pass

    if final_exit:
        raise SystemExit(final_exit)


if __name__ == "__main__":
    main()
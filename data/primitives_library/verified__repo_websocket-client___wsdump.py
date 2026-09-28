import argparse
import code
import gzip
import ssl
import sys
import threading
import time
import zlib
from urllib.parse import urlparse

import websocket

try:
    import readline
except ImportError:
    pass


OPCODE_DATA = (websocket.ABNF.OPCODE_TEXT, websocket.ABNF.OPCODE_BINARY)


class VAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.Namespace,
        args: tuple,
        values: str,
        option_string: str = None,
    ) -> None:
        if values is None:
            values = "1"
        try:
            level = int(values)
        except ValueError:
            level = values.count("v") + 1
        setattr(args, self.dest, level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket Simple Dump Tool")
    parser.add_argument(
        "url",
        metavar="ws_url",
        help="websocket url. ex. ws://websockets.chilkat.io/wsChilkatEcho.ashx",
    )
    parser.add_argument("-p", "--proxy", help="proxy url. ex. http://127.0.0.1:8080")
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        nargs="?",
        action=VAction,
        dest="verbose",
        help="set verbose mode. If set to 1, show opcode. "
        "If set to 2, enable to trace  websocket module",
    )
    parser.add_argument(
        "-n", "--nocert", action="store_true", help="Ignore invalid SSL cert"
    )
    parser.add_argument("-r", "--raw", action="store_true", help="raw output")
    parser.add_argument("-s", "--subprotocols", nargs="*", help="Set subprotocols")
    parser.add_argument("-o", "--origin", help="Set origin")
    parser.add_argument(
        "--eof-wait",
        default=0,
        type=int,
        help="wait time(second) after 'EOF' received.",
    )
    parser.add_argument("-t", "--text", help="Send initial text")
    parser.add_argument(
        "--timings", action="store_true", help="Print timings in seconds"
    )
    parser.add_argument("--headers", help="Set custom headers. Use ',' as separator")
    return parser.parse_args()


class RawInput:
    def raw_input(self, prompt: str = "") -> bytes:
        return input(prompt).encode("utf-8")


class InteractiveConsole(RawInput, code.InteractiveConsole):
    def write(self, data: str) -> None:
        sys.stdout.write("\033[2K\033[E")
        sys.stdout.write("\033[34m< " + data + "\033[39m")
        sys.stdout.write("\n> ")
        sys.stdout.flush()

    def read(self) -> bytes:
        return self.raw_input("> ")


class NonInteractive(RawInput):
    def write(self, data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def read(self) -> bytes:
        return self.raw_input("")


def main() -> None:
    start_time = time.time()
    args = parse_args()

    if args.verbose > 1:
        websocket.enableTrace(True)

    options = {}
    if args.proxy:
        parsed_proxy = urlparse(args.proxy)
        options["http_proxy_host"] = parsed_proxy.hostname
        options["http_proxy_port"] = parsed_proxy.port
    if args.origin:
        options["origin"] = args.origin
    if args.subprotocols:
        options["subprotocols"] = args.subprotocols

    ssl_options = {}
    if args.nocert:
        ssl_options = {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}

    if args.headers:
        options["header"] = list(map(str.strip, args.headers.split(",")))

    ws = websocket.create_connection(args.url, sslopt=ssl_options, **options)

    if args.raw:
        console = NonInteractive()
    else:
        console = InteractiveConsole()
        print("Press Ctrl+C to quit")

    def recv() -> tuple:
        try:
            frame = ws.recv_frame()
        except websocket.WebSocketException:
            return websocket.ABNF.OPCODE_CLOSE, ""

        if frame.opcode in OPCODE_DATA:
            return frame.opcode, frame.data
        if frame.opcode == websocket.ABNF.OPCODE_CLOSE:
            ws.send_close()
            return frame.opcode, ""
        if frame.opcode == websocket.ABNF.OPCODE_PING:
            ws.pong(frame.data)
            return frame.opcode, frame.data
        return frame.opcode, frame.data

    def recv_ws() -> None:
        while True:
            opcode, data = recv()

            if opcode == websocket.ABNF.OPCODE_TEXT and isinstance(data, bytes):
                data = str(data, "utf-8")

            if isinstance(data, bytes) and len(data) > 2 and data[:2] == b"\037\213":
                try:
                    data = "[gzip] " + str(gzip.decompress(data), "utf-8")
                except BaseException:
                    pass
            elif isinstance(data, bytes):
                try:
                    data = "[zlib] " + str(
                        zlib.decompress(data, -zlib.MAX_WBITS), "utf-8"
                    )
                except BaseException:
                    pass

            if isinstance(data, bytes):
                data = repr(data)

            if args.verbose:
                message = f"{websocket.ABNF.OPCODE_MAP.get(opcode)}: {data}"
            else:
                message = data

            if message is not None:
                if args.timings:
                    console.write(f"{time.time() - start_time}: {message}")
                else:
                    console.write(message)

            if opcode == websocket.ABNF.OPCODE_CLOSE:
                break

    thread = threading.Thread(target=recv_ws)
    thread.daemon = True
    thread.start()

    if args.text:
        ws.send(args.text)

    while True:
        try:
            message = console.read()
            ws.send(message)
        except KeyboardInterrupt:
            return
        except EOFError:
            time.sleep(args.eof_wait)
            return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
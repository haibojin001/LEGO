import asyncio
import dataclasses
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Union

from .lsp_requests import LspNotification, LspRequest
from .lsp_types import ErrorCodes

StringDict = Dict[str, Any]
PayloadLike = Union[List[StringDict], StringDict, None]
CONTENT_LENGTH = "Content-Length: "
ENCODING = "utf-8"


@dataclasses.dataclass
class ProcessLaunchInfo:
    cmd: str
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    cwd: str = os.getcwd()


class Error(Exception):
    def __init__(self, code: ErrorCodes, message: str) -> None:
        super().__init__(message)
        self.code = code

    def to_lsp(self) -> StringDict:
        return {"code": self.code, "message": super().__str__()}

    @classmethod
    def from_lsp(cls, d: StringDict) -> "Error":
        return cls(d["code"], d["message"])

    def __str__(self) -> str:
        return "{} ({})".format(super().__str__(), self.code)


def make_response(request_id: Any, params: PayloadLike) -> StringDict:
    return {"jsonrpc": "2.0", "id": request_id, "result": params}


def make_error_response(request_id: Any, err: Error) -> StringDict:
    return {"jsonrpc": "2.0", "id": request_id, "error": err.to_lsp()}


def make_notification(method: str, params: PayloadLike) -> StringDict:
    return {"jsonrpc": "2.0", "method": method, "params": params}


def make_request(method: str, request_id: Any, params: PayloadLike) -> StringDict:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "id": request_id,
        "params": params,
    }


class StopLoopException(Exception):
    pass


def create_message(payload: PayloadLike):
    body = json.dumps(
        payload,
        check_circular=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode(ENCODING)
    return (
        "{}{}\r\n".format(CONTENT_LENGTH, len(body)).encode(ENCODING),
        b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n",
        body,
    )


class MessageType:
    error = 1
    warning = 2
    info = 3
    log = 4


class Request:
    def __init__(self) -> None:
        self.cv = asyncio.Condition()
        self.result: Optional[PayloadLike] = None
        self.error: Optional[Error] = None

    async def on_result(self, params: PayloadLike) -> None:
        self.result = params
        async with self.cv:
            self.cv.notify()

    async def on_error(self, err: Error) -> None:
        self.error = err
        async with self.cv:
            self.cv.notify()


def content_length(line: bytes) -> Optional[int]:
    if line.startswith(b"Content-Length: "):
        _, value = line.split(b"Content-Length: ")
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            raise ValueError("Invalid Content-Length header: {}".format(value))
    return None


class LanguageServerHandler:
    def __init__(self, process_launch_info: ProcessLaunchInfo, logger=None) -> None:
        self.send = LspRequest(self.send_request)
        self.notify = LspNotification(self.send_notification)

        self.process_launch_info = process_launch_info
        self.process = None
        self._received_shutdown = False

        self.request_id = 1
        self._response_handlers: Dict[Any, Request] = {}
        self.on_request_handlers = {}
        self.on_notification_handlers = {}
        self.logger = logger
        self.tasks = {}
        self.task_counter = 0
        self.loop = None

    async def start(self) -> None:
        child_proc_env = os.environ.copy()
        child_proc_env.update(self.process_launch_info.env)

        self.process = await asyncio.create_subprocess_shell(
            self.process_launch_info.cmd,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_proc_env,
            cwd=self.process_launch_info.cwd,
        )

        self.loop = asyncio.get_event_loop()
        self.tasks[self.task_counter] = self.loop.create_task(self.run_forever())
        self.task_counter += 1
        self.tasks[self.task_counter] = self.loop.create_task(
            self.run_forever_stderr()
        )
        self.task_counter += 1

    async def stop(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        self.tasks = {}

        process = self.process
        self.process = None
        if process is None:
            return

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass

    def on_request(self, method: str):
        def register(handler):
            self.on_request_handlers[method] = handler
            return handler

        return register

    def on_notification(self, method: str):
        def register(handler):
            self.on_notification_handlers[method] = handler
            return handler

        return register

    def register_request_handler(self, method: str, handler) -> None:
        self.on_request_handlers[method] = handler

    def register_notification_handler(self, method: str, handler) -> None:
        self.on_notification_handlers[method] = handler

    async def write(self, payload: PayloadLike) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Language server process is not running")

        if self.logger is not None:
            self.logger("client", "server", payload)

        for chunk in create_message(payload):
            self.process.stdin.write(chunk)
        await self.process.stdin.drain()

    async def send_request(self, method: str, params: PayloadLike = None) -> PayloadLike:
        if self._received_shutdown:
            raise RuntimeError("Language server has been shut down")

        request_id = self.request_id
        self.request_id += 1

        request = Request()
        self._response_handlers[request_id] = request

        try:
            await self.write(make_request(method, request_id, params))
            async with request.cv:
                while request.result is None and request.error is None:
                    await request.cv.wait()

            if request.error is not None:
                raise request.error
            return request.result
        finally:
            self._response_handlers.pop(request_id, None)

    async def send_response(self, request_id: Any, params: PayloadLike = None) -> None:
        await self.write(make_response(request_id, params))

    async def send_error_response(self, request_id: Any, err: Error) -> None:
        await self.write(make_error_response(request_id, err))

    async def send_notification(
        self, method: str, params: PayloadLike = None
    ) -> None:
        if self._received_shutdown:
            return
        await self.write(make_notification(method, params))

    async def run_forever_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return

        while True:
            line = await self.process.stderr.readline()
            if not line:
                return
            try:
                print(line.decode(ENCODING), end="")
            except UnicodeDecodeError:
                print(line.decode(ENCODING, errors="replace"), end="")

    async def run_forever(self) -> None:
        if self.process is None or self.process.stdout is None:
            return

        try:
            while True:
                length = None
                while True:
                    line = await self.process.stdout.readline()
                    if not line:
                        raise StopLoopException()
                    if line in (b"\r\n", b"\n"):
                        break
                    parsed_length = content_length(line)
                    if parsed_length is not None:
                        length = parsed_length

                if length is None:
                    continue

                body = await self.process.stdout.readexactly(length)
                message = json.loads(body.decode(ENCODING))
                await self.handle_message(message)
        except StopLoopException:
            return
        except asyncio.IncompleteReadError:
            return

    async def handle_message(self, message) -> None:
        if isinstance(message, list):
            for item in message:
                await self.handle_message(item)
            return

        if not isinstance(message, dict):
            return

        if self.logger is not None:
            self.logger("server", "client", message)

        if "method" in message:
            if "id" in message:
                await self.handle_request(message)
            else:
                await self.handle_notification(message)
            return

        if "id" in message:
            await self.handle_response(message)

    async def handle_response(self, message: StringDict) -> None:
        request = self._response_handlers.get(message.get("id"))
        if request is None:
            return

        if "error" in message:
            await request.on_error(Error.from_lsp(message["error"]))
        else:
            await request.on_result(message.get("result"))

    async def handle_request(self, message: StringDict) -> None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params")

        if method == "shutdown":
            self._received_shutdown = True

        handler = self.on_request_handlers.get(method)
        if handler is None:
            code = getattr(ErrorCodes, "MethodNotFound", -32601)
            await self.send_error_response(
                request_id, Error(code, "Method not found: {}".format(method))
            )
            return

        try:
            result = handler(params)
            if inspect.isawaitable(result):
                result = await result
            await self.send_response(request_id, result)
        except Error as err:
            await self.send_error_response(request_id, err)
        except Exception as exc:
            code = getattr(ErrorCodes, "InternalError", -32603)
            await self.send_error_response(request_id, Error(code, str(exc)))

    async def handle_notification(self, message: StringDict) -> None:
        handler = self.on_notification_handlers.get(message.get("method"))
        if handler is None:
            return

        result = handler(message.get("params"))
        if inspect.isawaitable(result):
            await result
import asyncio
import dataclasses
import inspect
import logging
import pathlib
from contextlib import asynccontextmanager, contextmanager
from pathlib import PurePath
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from .lsp_protocol_handler.lsp_constants import LSPConstants
from .lsp_protocol_handler import lsp_types as LSPTypes
from . import multilspy_types
from .multilspy_logger import MultilspyLogger
from .lsp_protocol_handler.server import LanguageServerHandler, ProcessLaunchInfo
from .multilspy_config import MultilspyConfig, Language
from .multilspy_exceptions import MultilspyException
from .multilspy_utils import FileUtils


@dataclasses.dataclass
class LSPFileBuffer:
    uri: str
    contents: str
    version: int
    language_id: str
    ref_count: int


class LanguageServer:
    @classmethod
    def create(
        cls, config: MultilspyConfig, logger: MultilspyLogger, repository_root_path: str
    ) -> "LanguageServer":
        if config.code_language == Language.PYTHON:
            from cover_agent.lsp_logic.multilspy.language_servers.jedi_language_server.jedi_server import (
                JediServer,
            )

            return JediServer(config, logger, repository_root_path)

        logger.log(f"Language {config.code_language} is not supported", logging.ERROR)
        raise MultilspyException(f"Language {config.code_language} is not supported")

    def __init__(
        self,
        config: MultilspyConfig,
        logger: MultilspyLogger,
        repository_root_path: str,
        process_launch_info: ProcessLaunchInfo,
        language_id: str,
    ):
        if type(self) is LanguageServer:
            raise MultilspyException(
                "LanguageServer is an abstract class and cannot be instantiated directly. "
                "Use LanguageServer.create method instead."
            )

        self.logger = logger
        self.server_started = False
        self.repository_root_path = repository_root_path
        self.completions_available = asyncio.Event()

        if config.trace_lsp_communication:

            def logging_fn(source: str, target: str, message: Any) -> None:
                self.logger.log(
                    f"LSP: {source} -> {target}: {str(message)}", logging.DEBUG
                )

        else:

            def logging_fn(source: str, target: str, message: Any) -> None:
                return None

        self.server = LanguageServerHandler(process_launch_info, logger=logging_fn)
        self.language_id = language_id
        self.open_file_buffers: Dict[str, LSPFileBuffer] = {}

    @asynccontextmanager
    async def start_server(self) -> AsyncIterator["LanguageServer"]:
        self.server_started = True
        try:
            yield self
        finally:
            self.server_started = False

    def _ensure_started(self, operation: str) -> None:
        if self.server_started:
            return
        self.logger.log(
            f"{operation} called before Language Server started", logging.ERROR
        )
        raise MultilspyException("Language Server not started")

    def _absolute_path(self, relative_file_path: str) -> str:
        return str(PurePath(self.repository_root_path, relative_file_path))

    def _uri(self, relative_file_path: str) -> str:
        return pathlib.Path(self._absolute_path(relative_file_path)).as_uri()

    def _position(self, line: int, column: int) -> Dict[str, int]:
        return {"line": line, "character": column}

    def _text_document_position(
        self, relative_file_path: str, line: int, column: int
    ) -> Dict[str, Any]:
        return {
            "textDocument": {"uri": self._uri(relative_file_path)},
            "position": self._position(line, column),
        }

    async def _request(self, names: Union[str, List[str]], params: Dict[str, Any]) -> Any:
        request = self.server.request
        candidates = [names] if isinstance(names, str) else names

        for name in candidates:
            method = getattr(request, name, None)
            if method is None:
                continue
            result = method(params)
            if inspect.isawaitable(result):
                return await result
            return result

        raise AttributeError(
            f"The language server request handler does not support any of: {candidates}"
        )

    @contextmanager
    def open_file(self, relative_file_path: str) -> Iterator[None]:
        self._ensure_started("open_file")

        absolute_file_path = self._absolute_path(relative_file_path)
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri in self.open_file_buffers:
            buffer = self.open_file_buffers[uri]
            assert buffer.uri == uri
            assert buffer.ref_count >= 1
            buffer.ref_count += 1
            try:
                yield
            finally:
                buffer.ref_count -= 1
        else:
            contents = FileUtils.read_file(self.logger, absolute_file_path)
            buffer = LSPFileBuffer(uri, contents, 0, self.language_id, 1)
            self.open_file_buffers[uri] = buffer

            self.server.notify.did_open_text_document(
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": self.language_id,
                        "version": 0,
                        "text": contents,
                    }
                }
            )

            try:
                yield
            finally:
                buffer.ref_count -= 1

        if uri in self.open_file_buffers and self.open_file_buffers[uri].ref_count == 0:
            self.server.notify.did_close_text_document(
                {"textDocument": {"uri": uri}}
            )
            del self.open_file_buffers[uri]

    def _buffer_for_file(self, relative_file_path: str) -> LSPFileBuffer:
        uri = self._uri(relative_file_path)
        assert uri in self.open_file_buffers
        return self.open_file_buffers[uri]

    def _notify_change(
        self,
        buffer: LSPFileBuffer,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        text: str,
    ) -> None:
        buffer.version += 1
        self.server.notify.did_change_text_document(
            {
                "textDocument": {
                    "uri": buffer.uri,
                    "version": buffer.version,
                },
                "contentChanges": [
                    {
                        "range": {
                            "start": self._position(start_line, start_column),
                            "end": self._position(end_line, end_column),
                        },
                        "text": text,
                    }
                ],
            }
        )

    @staticmethod
    def _offset(contents: str, line: int, column: int) -> int:
        lines = contents.splitlines(keepends=True)
        if line < 0 or column < 0:
            raise IndexError("Line and column must be non-negative")

        if line > len(lines):
            raise IndexError("Line index out of range")

        if line == len(lines):
            if column != 0:
                raise IndexError("Column index out of range")
            return len(contents)

        line_text = lines[line]
        logical_line = line_text.rstrip("\r\n")
        if column > len(logical_line):
            raise IndexError("Column index out of range")

        return sum(len(item) for item in lines[:line]) + column

    def insert_text_at_position(
        self, relative_file_path: str, line: int, column: int, text_to_be_inserted: str
    ) -> multilspy_types.Position:
        self._ensure_started("insert_text_at_position")
        buffer = self._buffer_for_file(relative_file_path)

        offset = self._offset(buffer.contents, line, column)
        buffer.contents = (
            buffer.contents[:offset] + text_to_be_inserted + buffer.contents[offset:]
        )
        self._notify_change(
            buffer,
            line,
            column,
            line,
            column,
            text_to_be_inserted,
        )

        newline_count = text_to_be_inserted.count("\n")
        if newline_count:
            return multilspy_types.Position(
                line + newline_count,
                len(text_to_be_inserted.rsplit("\n", 1)[-1]),
            )
        return multilspy_types.Position(line, column + len(text_to_be_inserted))

    def delete_text_between_positions(
        self,
        relative_file_path: str,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
    ) -> None:
        self._ensure_started("delete_text_between_positions")
        buffer = self._buffer_for_file(relative_file_path)

        start = self._offset(buffer.contents, start_line, start_column)
        end = self._offset(buffer.contents, end_line, end_column)
        if end < start:
            raise ValueError("End position must not precede start position")

        buffer.contents = buffer.contents[:start] + buffer.contents[end:]
        self._notify_change(
            buffer,
            start_line,
            start_column,
            end_line,
            end_column,
            "",
        )

    async def request_hover(
        self, relative_file_path: str, line: int, column: int
    ) -> Any:
        self._ensure_started("request_hover")
        return await self._request(
            "hover", self._text_document_position(relative_file_path, line, column)
        )

    async def request_definition(
        self, relative_file_path: str, line: int, column: int
    ) -> Any:
        self._ensure_started("request_definition")
        return await self._request(
            ["definition", "goto_definition"],
            self._text_document_position(relative_file_path, line, column),
        )

    async def request_declaration(
        self, relative_file_path: str, line: int, column: int
    ) -> Any:
        self._ensure_started("request_declaration")
        return await self._request(
            ["declaration", "goto_declaration"],
            self._text_document_position(relative_file_path, line, column),
        )

    async def request_type_definition(
        self, relative_file_path: str, line: int, column: int
    ) -> Any:
        self._ensure_started("request_type_definition")
        return await self._request(
            ["type_definition", "typeDefinition"],
            self._text_document_position(relative_file_path, line, column),
        )

    async def request_references(
        self,
        relative_file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
    ) -> Any:
        self._ensure_started("request_references")
        params = self._text_document_position(relative_file_path, line, column)
        params["context"] = {"includeDeclaration": include_declaration}
        return await self._request("references", params)

    async def request_completions(
        self,
        relative_file_path: str,
        line: int,
        column: int,
    ) -> Any:
        self._ensure_started("request_completions")
        params = self._text_document_position(relative_file_path, line, column)
        return await self._request(["completion", "completions"], params)

    async def request_signature_help(
        self, relative_file_path: str, line: int, column: int
    ) -> Any:
        self._ensure_started("request_signature_help")
        return await self._request(
            ["signature_help", "signatureHelp"],
            self._text_document_position(relative_file_path, line, column),
        )

    async def request_document_symbols(self, relative_file_path: str) -> Any:
        self._ensure_started("request_document_symbols")
        return await self._request(
            ["document_symbol", "document_symbols"],
            {"textDocument": {"uri": self._uri(relative_file_path)}},
        )

    async def request_workspace_symbols(self, query: str = "") -> Any:
        self._ensure_started("request_workspace_symbols")
        return await self._request(
            ["workspace_symbol", "workspace_symbols"], {"query": query}
        )
import json
import logging
import os
import pathlib
from contextlib import asynccontextmanager
from typing import AsyncIterator

from cover_agent.lsp_logic.multilspy.language_server import LanguageServer
from cover_agent.lsp_logic.multilspy.lsp_protocol_handler.lsp_types import (
    InitializeParams,
)
from cover_agent.lsp_logic.multilspy.lsp_protocol_handler.server import (
    ProcessLaunchInfo,
)
from cover_agent.lsp_logic.multilspy.multilspy_config import MultilspyConfig
from cover_agent.lsp_logic.multilspy.multilspy_logger import MultilspyLogger


class JediServer(LanguageServer):
    """
    Python-specific language server implementation backed by jedi-language-server.
    """

    def __init__(
        self,
        config: MultilspyConfig,
        logger: MultilspyLogger,
        repository_root_path: str,
    ):
        """
        Create a Jedi language-server wrapper.

        Instances are normally created through LanguageServer.create().
        """
        launch_info = ProcessLaunchInfo(
            cmd="jedi-language-server",
            cwd=repository_root_path,
        )
        super().__init__(
            config,
            logger,
            repository_root_path,
            launch_info,
            "python",
        )

    def _get_initialize_params(
        self,
        repository_absolute_path: str,
    ) -> InitializeParams:
        """
        Build initialization parameters from the bundled Jedi configuration.
        """
        configuration_path = os.path.join(
            os.path.dirname(__file__),
            "initialize_params.json",
        )
        with open(configuration_path, "r") as configuration_file:
            params = json.load(configuration_file)

        del params["_description"]

        params["processId"] = os.getpid()

        assert params["rootPath"] == "$rootPath"
        params["rootPath"] = repository_absolute_path

        repository_uri = pathlib.Path(repository_absolute_path).as_uri()

        assert params["rootUri"] == "$rootUri"
        params["rootUri"] = repository_uri

        workspace_folder = params["workspaceFolders"][0]

        assert workspace_folder["uri"] == "$uri"
        workspace_folder["uri"] = repository_uri

        assert workspace_folder["name"] == "$name"
        workspace_folder["name"] = os.path.basename(repository_absolute_path)

        return params

    @asynccontextmanager
    async def start_server(self) -> AsyncIterator["JediServer"]:
        """
        Start Jedi, initialize its LSP session, and shut it down on context exit.
        """

        async def execute_client_command_handler(params):
            return []

        async def do_nothing(params):
            return

        async def check_experimental_status(params):
            if params["quiescent"] == True:
                self.completions_available.set()

        async def window_log_message(msg):
            self.logger.log(f"LSP: window/logMessage: {msg}", logging.INFO)

        self.server.on_request("client/registerCapability", do_nothing)
        self.server.on_notification("language/status", do_nothing)
        self.server.on_notification("window/logMessage", window_log_message)
        self.server.on_request(
            "workspace/executeClientCommand",
            execute_client_command_handler,
        )
        self.server.on_notification("$/progress", do_nothing)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)
        self.server.on_notification("language/actionableNotification", do_nothing)
        self.server.on_notification(
            "experimental/serverStatus",
            check_experimental_status,
        )

        async with super().start_server():
            self.logger.log(
                "Starting jedi-language-server server process",
                logging.INFO,
            )
            await self.server.start()

            initialize_params = self._get_initialize_params(
                self.repository_root_path,
            )

            self.logger.log(
                "Sending initialize request from LSP client to LSP server and awaiting response",
                logging.INFO,
            )
            init_response = await self.server.send.initialize(initialize_params)

            capabilities = init_response["capabilities"]
            assert capabilities["textDocumentSync"]["change"] == 2
            assert "completionProvider" in capabilities
            assert capabilities["completionProvider"] == {
                "triggerCharacters": [".", "'", '"'],
                "resolveProvider": True,
            }

            self.server.notify.initialized({})

            yield self

            await self.server.shutdown()
            await self.server.stop()
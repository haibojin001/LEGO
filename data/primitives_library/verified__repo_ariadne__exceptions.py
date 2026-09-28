import os

from .constants import HttpStatusResponse


class HttpError(Exception):
    """Base exception for HTTP failures produced by server integrations."""

    def __init__(self, status: str, message: str | None = None) -> None:
        super().__init__()
        self.status = status
        self.message = message


class HttpBadRequestError(HttpError):
    """Exception raised when a request lacks required GraphQL input data."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(HttpStatusResponse.BAD_REQUEST.value, message)


class GraphQLFileSyntaxError(Exception):
    """Exception raised when a GraphQL schema file cannot be parsed."""

    def __init__(self, file_path: str | os.PathLike, message: str) -> None:
        super().__init__()
        self.message = self.format_message(file_path, message)

    def format_message(self, file_path: str | os.PathLike, message: str):
        return f"Could not load {file_path}:\n{message}"

    def __str__(self):
        return self.message


class WebSocketConnectionError(Exception):
    """Exception used to provide custom WebSocket connection failure payloads."""

    def __init__(self, payload: dict | str | None = None) -> None:
        if isinstance(payload, dict):
            self.payload = payload
        elif payload:
            self.payload = {"message": str(payload)}
        else:
            self.payload = {"message": "Unexpected error has occurred."}
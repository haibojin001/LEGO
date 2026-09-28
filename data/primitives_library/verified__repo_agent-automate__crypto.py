from __future__ import annotations

import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet


class Vault:
    def __init__(self, key_path: Path):
        self._path = key_path
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self._path.exists():
            return self._path.read_bytes()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        self._path.write_bytes(key)

        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass

        return key

    def encrypt(self, plaintext: str) -> str:
        if plaintext is None:
            return ""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext:
            return ""
        return self._fernet.decrypt(ciphertext.encode()).decode()

    @staticmethod
    def random_token(nbytes: int = 32) -> str:
        return secrets.token_urlsafe(nbytes)
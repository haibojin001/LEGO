from __future__ import annotations

import hashlib
import os
import sys
import typing as t
from base64 import encodebytes
from collections import OrderedDict
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path

try:
    import sqlite3

    def adapt_datetime_iso(val):
        return val.isoformat()

    def convert_datetime(val):
        return datetime.fromisoformat(val.decode())

    sqlite3.register_adapter(datetime, adapt_datetime_iso)
    sqlite3.register_converter("datetime", convert_datetime)
except ImportError:
    try:
        from pysqlite2 import dbapi2 as sqlite3  # type: ignore[no-redef]
    except ImportError:
        sqlite3 = None  # type: ignore[assignment]

from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError

from . import NO_CONVERT, __version__, read, reads

algorithms_set = hashlib.algorithms_guaranteed
algorithms = [algorithm for algorithm in algorithms_set if not algorithm.startswith("shake_")]


class SignatureStore:
    """Base class for a signature store."""

    def store_signature(self, digest, algorithm):
        """Store a signature."""
        raise NotImplementedError

    def check_signature(self, digest, algorithm):
        """Check whether a signature is stored."""
        raise NotImplementedError

    def remove_signature(self, digest, algorithm):
        """Remove a signature."""
        raise NotImplementedError

    def close(self):
        """Close resources used by the store."""


class MemorySignatureStore(SignatureStore):
    """A non-persistent, in-memory signature store."""

    cache_size = 65535

    def __init__(self):
        self.data = OrderedDict()

    def store_signature(self, digest, algorithm):
        key = (digest, algorithm)
        self.data.pop(key, None)
        self.data[key] = None
        self._maybe_cull()

    def _maybe_cull(self):
        if len(self.data) < self.cache_size:
            return
        for _ in range(len(self.data) // 4):
            self.data.popitem(last=False)

    def check_signature(self, digest, algorithm):
        key = (digest, algorithm)
        if key not in self.data:
            return False
        del self.data[key]
        self.data[key] = None
        return True

    def remove_signature(self, digest, algorithm):
        self.data.pop((digest, algorithm), None)


class SQLiteSignatureStore(SignatureStore, LoggingConfigurable):
    """A signature store backed by an SQLite database."""

    cache_size = Integer(
        65535,
        help=(
            "The number of notebook signatures to cache. When this number is "
            "exceeded, the oldest quarter of signatures is removed."
        ),
    ).tag(config=True)

    def __init__(self, db_file, **kwargs):
        super().__init__(**kwargs)
        self.db_file = db_file
        self.db = self._connect_db(db_file)

    def close(self):
        if self.db is not None:
            self.db.close()

    def _connect_db(self, db_file):
        kwargs: dict[str, t.Any] = {
            "detect_types": sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        }
        db = None
        try:
            db = sqlite3.connect(db_file, **kwargs)
            self.init_db(db)
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            if db_file == ":memory:":
                raise

            backup = db_file + ".bak"
            if db is not None:
                db.close()

            self.log.warning(
                "The signatures database cannot be opened; maybe it is corrupted "
                "or encrypted. You may need to rerun your notebooks to ensure that "
                "they are trusted to run Javascript. The old signatures database "
                "has been renamed to %s and a new one has been created.",
                backup,
            )
            try:
                Path(db_file).rename(backup)
                db = sqlite3.connect(db_file, **kwargs)
                self.init_db(db)
            except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError):
                if db is not None:
                    db.close()
                self.log.warning(
                    "Failed committing signatures database to disk. You may need to "
                    "move the database file to a non-networked file system, using "
                    "config option `NotebookNotary.db_file`. Using in-memory "
                    "signatures database for the remainder of this session."
                )
                self.db_file = ":memory:"
                db = sqlite3.connect(":memory:", **kwargs)
                self.init_db(db)
        return db

    def init_db(self, db):
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS nbsignatures
            (
                id integer PRIMARY KEY AUTOINCREMENT,
                algorithm text,
                signature text,
                path text,
                last_seen timestamp
            )
            """
        )
        db.execute(
            """
            CREATE INDEX IF NOT EXISTS algosig
            ON nbsignatures(algorithm, signature)
            """
        )
        db.commit()

    def store_signature(self, digest, algorithm):
        if self.db is None:
            return

        now = datetime.now(tz=timezone.utc)
        if self.check_signature(digest, algorithm):
            self.db.execute(
                """
                UPDATE nbsignatures
                SET last_seen = ?
                WHERE algorithm = ? AND signature = ?
                """,
                (now, algorithm, digest),
            )
        else:
            self.db.execute(
                """
                INSERT INTO nbsignatures (algorithm, signature, last_seen)
                VALUES (?, ?, ?)
                """,
                (algorithm, digest, now),
            )
        self.db.commit()

        (count,) = self.db.execute("SELECT Count(*) FROM nbsignatures").fetchone()
        if count > self.cache_size:
            self.cull_db()

    def check_signature(self, digest, algorithm):
        if self.db is None:
            return False

        row = self.db.execute(
            """
            SELECT id FROM nbsignatures
            WHERE algorithm = ? AND signature = ?
            """,
            (algorithm, digest),
        ).fetchone()

        if row is None:
            return False

        self.db.execute(
            """
            UPDATE nbsignatures
            SET last_seen = ?
            WHERE algorithm = ? AND signature = ?
            """,
            (datetime.now(tz=timezone.utc), algorithm, digest),
        )
        self.db.commit()
        return True

    def remove_signature(self, digest, algorithm):
        if self.db is None:
            return
        self.db.execute(
            """
            DELETE FROM nbsignatures
            WHERE algorithm = ? AND signature = ?
            """,
            (algorithm, digest),
        )
        self.db.commit()

    def cull_db(self):
        if self.db is None:
            return

        (count,) = self.db.execute("SELECT Count(*) FROM nbsignatures").fetchone()
        if count < self.cache_size:
            return

        self.db.execute(
            """
            DELETE FROM nbsignatures
            WHERE id IN (
                SELECT id FROM nbsignatures
                ORDER BY last_seen ASC
                LIMIT ?
            )
            """,
            (count // 4,),
        )
        self.db.commit()


class NotebookNotary(LoggingConfigurable):
    """Compute, store, and verify signatures for notebooks."""

    data_dir = Unicode(help="The location of the Jupyter data directory.").tag(config=True)

    db_file = Unicode(
        help="The SQLite database file in which notebook signatures are stored."
    ).tag(config=True)

    secret_file = Unicode(
        help="The file containing the secret used to sign notebooks."
    ).tag(config=True)

    store_factory = Callable(help="A factory used to create the signature store.").tag(config=True)

    signature_store = Instance(SignatureStore, allow_none=True)

    algorithm = Enum(
        algorithms,
        default_value="sha256",
        help="The hash algorithm used to sign notebooks.",
    ).tag(config=True)

    digestmod = Any()

    secret = Bytes(help="The secret key used to sign notebooks.").tag(config=True)

    @default("data_dir")
    def _data_dir_default(self):
        try:
            app = JupyterApp.instance()
        except MultipleInstanceError:
            app = JupyterApp()
        return app.data_dir

    @default("db_file")
    def _db_file_default(self):
        return os.path.join(self.data_dir, "nbsignatures.db")

    @default("secret_file")
    def _secret_file_default(self):
        return os.path.join(self.data_dir, "notebook_secret")

    @default("store_factory")
    def _store_factory_default(self):
        return SQLiteSignatureStore

    @default("signature_store")
    def _signature_store_default(self):
        if sqlite3 is None:
            return MemorySignatureStore()
        return self.store_factory(self.db_file, parent=self)

    @default("digestmod")
    def _digestmod_default(self):
        return getattr(hashlib, self.algorithm)

    @observe("algorithm")
    def _algorithm_changed(self, change):
        self.digestmod = getattr(hashlib, change["new"])

    @default("secret")
    def _secret_default(self):
        try:
            with open(self.secret_file, "rb") as f:
                return f.read()
        except OSError:
            pass

        secret = encodebytes(os.urandom(1024))
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.secret_file, "wb") as f:
                f.write(secret)
            try:
                os.chmod(self.secret_file, 0o600)
            except OSError:
                pass
        except OSError:
            self.log.warning(
                "Could not create the notebook signing key. Notebook signatures "
                "will not persist between sessions."
            )
        return secret

    def _yield_everything(self, obj):
        if isinstance(obj, dict):
            for key in sorted(obj):
                yield from self._yield_everything(key)
                yield from self._yield_everything(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from self._yield_everything(item)
        elif isinstance(obj, str):
            yield obj.encode("utf8")
        else:
            yield repr(obj).encode("utf8")

    def compute_signature(self, nb):
        """Compute the HMAC digest for a notebook."""
        hmac = HMAC(self.secret, digestmod=self.digestmod)
        for chunk in self._yield_everything(nb):
            hmac.update(chunk)
        return hmac.hexdigest()

    def check_signature(self, nb):
        """Return whether the notebook has a known signature."""
        digest = self.compute_signature(nb)
        return self.signature_store.check_signature(digest, self.algorithm)

    def sign(self, nb):
        """Store the current notebook signature."""
        digest = self.compute_signature(nb)
        self.signature_store.store_signature(digest, self.algorithm)

    def unsign(self, nb):
        """Remove the current notebook signature from the signature store."""
        digest = self.compute_signature(nb)
        self.signature_store.remove_signature(digest, self.algorithm)

    def mark_cells(self, nb, trusted):
        """Mark all code cells in a notebook as trusted or untrusted."""
        if getattr(nb, "nbformat", 4) < 4:
            cells = (
                cell
                for worksheet in nb.get("worksheets", [])
                for cell in worksheet.get("cells", [])
            )
        else:
            cells = nb.get("cells", [])

        for cell in cells:
            if cell.get("cell_type") == "code":
                cell.setdefault("metadata", {})["trusted"] = trusted

    def _check_cell(self, cell, nb_version):
        if cell.get("cell_type") != "code":
            return True

        if cell.get("metadata", {}).get("trusted", False):
            return True

        if nb_version < 4:
            unsafe_output_types = {"pyout", "display_data"}
        else:
            unsafe_output_types = {"execute_result", "display_data"}

        for output in cell.get("outputs", []):
            if output.get("output_type") in unsafe_output_types:
                return False
        return True

    def check_cells(self, nb):
        """Return whether every code cell in the notebook is trusted."""
        version = nb.get("nbformat", 4)

        if version < 4:
            return all(
                self._check_cell(cell, version)
                for worksheet in nb.get("worksheets", [])
                for cell in worksheet.get("cells", [])
            )

        return all(self._check_cell(cell, version) for cell in nb.get("cells", []))


class TrustNotebookApp(JupyterApp):
    """Application for signing notebooks from the command line."""

    name = "jupyter trust"
    description = "Sign one or more Jupyter notebooks with your key."
    version = __version__

    flags = dict(base_flags)
    flags.update(
        {
            "reset": (
                {"TrustNotebookApp": {"reset": True}},
                "Reset the notebook trust database.",
            )
        }
    )

    notary = Instance(NotebookNotary)
    reset = Bool(False, help="Reset the notebook trust database.").tag(config=True)

    @default("notary")
    def _notary_default(self):
        return NotebookNotary(parent=self)

    def sign_notebook(self, notebook_path):
        if notebook_path == "-":
            nb = reads(sys.stdin.read(), NO_CONVERT)
        else:
            nb = read(notebook_path, NO_CONVERT)
        self.notary.sign(nb)

    def start(self):
        if self.reset:
            db_file = self.notary.db_file
            self.notary.close()
            try:
                os.remove(db_file)
            except OSError:
                pass
            return

        if not self.extra_args:
            self.log.error("Specify at least one notebook to sign.")
            self.exit(1)

        for notebook_path in self.extra_args:
            try:
                self.sign_notebook(notebook_path)
            except Exception:
                self.log.error("Error signing notebook: %s", notebook_path, exc_info=True)


main = TrustNotebookApp.launch_instance
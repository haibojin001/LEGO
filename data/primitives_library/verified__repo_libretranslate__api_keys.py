import os
import sqlite3
import uuid

import requests
from expiringdict import ExpiringDict

from libretranslate.default_values import DEFAULT_ARGUMENTS as DEFARGS


DEFAULT_DB_PATH = DEFARGS["API_KEYS_DB_PATH"]


class Database:
    def __init__(self, db_path=DEFAULT_DB_PATH, max_cache_len=1000, max_cache_age=30):
        legacy_path = "api_keys.db"
        migrated_path = "db/api_keys.db"

        if os.path.isfile(legacy_path) and not os.path.isfile(migrated_path):
            print("Migrating {} to {}".format(legacy_path, migrated_path))
            try:
                os.rename(legacy_path, migrated_path)
            except Exception as error:
                print(str(error))

        directory = os.path.dirname(db_path)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)

        self.db_path = db_path
        self.cache = ExpiringDict(
            max_len=max_cache_len,
            max_age_seconds=max_cache_age,
        )
        self.c = sqlite3.connect(db_path, check_same_thread=False)

        self.c.execute(
            """CREATE TABLE IF NOT EXISTS api_keys (
            "api_key" TEXT NOT NULL,
            "req_limit" INTEGER NOT NULL,
            "char_limit" INTEGER DEFAULT NULL,
            PRIMARY KEY("api_key")
        );"""
        )

        schema = self.c.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='api_keys';"
        ).fetchone()[0]

        if '"char_limit" INTEGER DEFAULT NULL' not in schema:
            self.c.execute(
                'ALTER TABLE api_keys ADD COLUMN "char_limit" INTEGER DEFAULT NULL;'
            )

    def lookup(self, api_key):
        value = self.cache.get(api_key)

        if value is None:
            cursor = self.c.execute(
                "SELECT req_limit, char_limit FROM api_keys WHERE api_key = ?",
                (api_key,),
            )
            result = cursor.fetchone()

            if result is None:
                value = False
            else:
                value = result

            self.cache[api_key] = value

        if isinstance(value, bool):
            value = None

        return value

    def add(self, req_limit, api_key="auto", char_limit=None):
        if api_key == "auto":
            api_key = str(uuid.uuid4())

        if char_limit == 0:
            char_limit = None

        self.remove(api_key)
        self.c.execute(
            "INSERT INTO api_keys (api_key, req_limit, char_limit) VALUES (?, ?, ?)",
            (api_key, req_limit, char_limit),
        )
        self.c.commit()

        return (api_key, req_limit, char_limit)

    def remove(self, api_key):
        self.c.execute(
            "DELETE FROM api_keys WHERE api_key = ?",
            (api_key,),
        )
        self.c.commit()
        return api_key

    def all(self):
        cursor = self.c.execute(
            "SELECT api_key, req_limit, char_limit FROM api_keys"
        )
        return cursor.fetchall()


class RemoteDatabase:
    def __init__(self, url, max_cache_len=1000, max_cache_age=600):
        self.url = url
        self.cache = ExpiringDict(
            max_len=max_cache_len,
            max_age_seconds=max_cache_age,
        )

    def lookup(self, api_key):
        value = self.cache.get(api_key)

        if value is None:
            try:
                response = requests.post(
                    self.url,
                    data={"api_key": api_key},
                    timeout=60,
                )
                result = response.json()
            except Exception as error:
                print("Cannot authenticate API key: " + str(error))
                return None

            if result.get("error") is not None:
                return None

            req_limit = result.get("req_limit", None)
            char_limit = result.get("char_limit", None)
            value = (req_limit, char_limit)
            self.cache[api_key] = value

        return value
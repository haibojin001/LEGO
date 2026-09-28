import os
import sqlite3

from expiringdict import ExpiringDict

DEFAULT_DB_PATH = "db/suggestions.db"


class Database:
    def __init__(self, db_path=DEFAULT_DB_PATH, max_cache_len=1000, max_cache_age=30):
        legacy_path = "suggestions.db"
        standard_path = "db/suggestions.db"

        if os.path.isfile(legacy_path) and not os.path.isfile(standard_path):
            print("Migrating {} to {}".format(legacy_path, standard_path))
            try:
                os.rename(legacy_path, standard_path)
            except Exception as exc:
                print(str(exc))

        self.db_path = db_path
        self.cache = ExpiringDict(
            max_len=max_cache_len,
            max_age_seconds=max_cache_age,
        )
        self.c = sqlite3.connect(db_path, check_same_thread=False)
        self.c.execute(
            """
            CREATE TABLE IF NOT EXISTS suggestions (
                "q" TEXT NOT NULL,
                "s" TEXT NOT NULL,
                "source" TEXT NOT NULL,
                "target" TEXT NOT NULL
            );
            """
        )

    def add(self, q, s, source, target):
        self.c.execute(
            "INSERT INTO suggestions (q, s, source, target) VALUES (?, ?, ?, ?)",
            (q, s, source, target),
        )
        self.c.commit()
        return True
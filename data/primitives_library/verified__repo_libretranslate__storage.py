import redis
import time

storage = None


def get_storage():
    return storage


class Storage:
    def exists(self, key):
        raise Exception("not implemented")

    def set_bool(self, key, value):
        raise Exception("not implemented")

    def get_bool(self, key):
        raise Exception("not implemented")

    def set_int(self, key, value):
        raise Exception("not implemented")

    def get_int(self, key):
        raise Exception("not implemented")

    def set_str(self, key, value, ex=None):
        raise Exception("not implemented")

    def get_str(self, key):
        raise Exception("not implemented")

    def set_hash_int(self, ns, key, value):
        raise Exception("not implemented")

    def get_hash_int(self, ns, key):
        raise Exception("not implemented")

    def inc_hash_int(self, ns, key):
        raise Exception("not implemented")

    def dec_hash_int(self, ns, key):
        raise Exception("not implemented")

    def get_hash_keys(self, ns):
        raise Exception("not implemented")

    def del_hash(self, ns, key):
        raise Exception("not implemented")


class MemoryStorage(Storage):
    def __init__(self):
        self.store = {}

    def exists(self, key):
        return key in self.store

    def set_bool(self, key, value):
        self.store[key] = bool(value)

    def get_bool(self, key):
        return bool(self.store[key])

    def set_int(self, key, value):
        self.store[key] = int(value)

    def get_int(self, key):
        return int(self.store.get(key, 0))

    def set_str(self, key, value, ex=None):
        self.store[key] = {
            "value": value,
            "ex": None if ex is None else time.time() + ex,
        }

    def get_str(self, key, raw=False):
        entry = self.store.get(key, {"value": "", "ex": None})
        if entry["ex"] is None:
            return entry["value"]
        if entry["ex"] <= time.time():
            del self.store[key]
            return ""
        return entry["value"]

    def set_hash_int(self, ns, key, value):
        if ns not in self.store:
            self.store[ns] = {}
        self.store[ns][key] = int(value)

    def get_hash_int(self, ns, key):
        return int(self.store.get(ns, {}).get(key, 0))

    def inc_hash_int(self, ns, key):
        if ns not in self.store:
            self.store[ns] = {}
        if key not in self.store[ns]:
            self.store[ns][key] = 0
        else:
            self.store[ns][key] += 1

    def dec_hash_int(self, ns, key):
        if ns not in self.store:
            self.store[ns] = {}
        if key not in self.store[ns]:
            self.store[ns][key] = 0
        else:
            self.store[ns][key] -= 1

    def get_all_hash_int(self, ns):
        if ns in self.store:
            return [{str(k): int(v)} for k, v in self.store[ns].items()]
        return []

    def del_hash(self, ns, key):
        del self.store[ns][key]


class RedisStorage(Storage):
    def __init__(self, redis_uri):
        self.conn = redis.from_url(redis_uri)
        self.conn.ping()

    def exists(self, key):
        return bool(self.conn.exists(key))

    def set_bool(self, key, value):
        self.conn.set(key, "1" if value else "0")

    def get_bool(self, key):
        return bool(self.conn.get(key))

    def set_int(self, key, value):
        self.conn.set(key, str(value))

    def get_int(self, key):
        value = self.conn.get(key)
        if value is None:
            return 0
        return value

    def set_str(self, key, value, ex=None):
        self.conn.set(key, value, ex=ex)

    def get_str(self, key, raw=False):
        value = self.conn.get(key)
        if value is None:
            return ""
        if raw:
            return value
        return value.decode("utf-8")

    def get_hash_int(self, ns, key):
        value = self.conn.hget(ns, key)
        if value is None:
            return 0
        return int(value)

    def set_hash_int(self, ns, key, value):
        self.conn.hset(ns, key, value)

    def inc_hash_int(self, ns, key):
        return int(self.conn.hincrby(ns, key))

    def dec_hash_int(self, ns, key):
        return int(self.conn.hincrby(ns, key, -1))

    def get_all_hash_int(self, ns):
        return {
            key.decode("utf-8"): int(value)
            for key, value in self.conn.hgetall(ns).items()
        }

    def del_hash(self, ns, key):
        self.conn.hdel(ns, key)


def setup(storage_uri):
    global storage

    if storage_uri.startswith("memory://"):
        storage = MemoryStorage()
    elif storage_uri.startswith("redis://"):
        storage = RedisStorage(storage_uri)
    else:
        raise Exception("Invalid storage URI: " + storage_uri)

    return storage
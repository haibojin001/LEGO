from libretranslate.storage import get_storage

active = False
threshold = -1


def forgive_banned():
    global threshold

    storage = get_storage()
    entries = storage.get_all_hash_int("banned")
    expired = []

    for address, count in entries.items():
        if count <= 0:
            expired.append(address)
        else:
            storage.set_hash_int("banned", address, min(threshold, count) - 1)

    for address in expired:
        storage.del_hash("banned", address)


def setup(args):
    global active, threshold

    value = args.req_flood_threshold
    if value > 0:
        active = True
        threshold = value


def report(request_ip):
    if active:
        get_storage().inc_hash_int("banned", request_ip)


def decrease(request_ip):
    storage = get_storage()
    if storage.get_hash_int("banned", request_ip) > 0:
        storage.dec_hash_int("banned", request_ip)


def has_violation(request_ip):
    return get_storage().get_hash_int("banned", request_ip) > 0


def is_banned(request_ip):
    return active and get_storage().get_hash_int("banned", request_ip) >= threshold


def fingerprint_mismatch(request_ip, fingerprint):
    if not isinstance(fingerprint, str) or not fingerprint:
        return True

    storage = get_storage()
    key = "fingerprint:" + str(request_ip)
    saved = storage.get_str(key)

    if saved == "":
        storage.set_str(key, fingerprint, ex=300)
        return False

    return fingerprint != saved
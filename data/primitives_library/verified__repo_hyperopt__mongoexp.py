import copy
import logging
import optparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import warnings

import numpy

try:
    import gridfs
    import pymongo
    from bson import SON
    try:
        from pymongo import ReturnDocument
    except Exception:
        ReturnDocument = None
    _has_mongo = True
except Exception:
    gridfs = None
    pymongo = None
    SON = dict
    ReturnDocument = None
    _has_mongo = False

from .base import (
    JOB_STATE_DONE,
    JOB_STATE_ERROR,
    JOB_STATE_NEW,
    JOB_STATE_RUNNING,
    JOB_STATES,
    Ctrl,
    InvalidTrial,
    SONify,
    Trials,
    spec_from_misc,
)
from .utils import coarse_utcnow, fast_isin, get_most_recent_inds, json_call, temp_dir, working_dir

__authors__ = ["James Bergstra", "Dan Yamins"]
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

logger = logging.getLogger(__name__)

try:
    import cloudpickle as pickler
except Exception:
    import pickle as pickler


class OperationFailure(Exception):
    pass


class Shutdown(Exception):
    pass


class WaitQuit(Exception):
    pass


class InvalidMongoTrial(InvalidTrial):
    pass


class DomainSwapError(Exception):
    pass


class ReserveTimeout(Exception):
    pass


def read_pw():
    with open(os.path.join(os.getenv("HOME"), ".hyperopt")) as f:
        return f.read()[:-1]


def parse_url(url, pwfile=None):
    protocol = url[:url.find(":")]
    parsed = urllib.parse.urlparse("ftp" + url[url.find(":"):])
    query = urllib.parse.parse_qs(parsed.query)

    try:
        _, dbname, collection = parsed.path.split("/")
    except Exception:
        print("Failed to parse '%s'" % parsed.path, file=sys.stderr)
        raise

    password = parsed.password
    if password is None and parsed.username is not None and pwfile:
        with open(pwfile) as f:
            password = f.read()[:-1]

    authdbname = None
    if query.get("authSource"):
        authdbname = query["authSource"][-1]

    return (
        protocol,
        parsed.username,
        password,
        parsed.hostname,
        int(float(parsed.port)),
        dbname,
        collection,
        authdbname,
    )


def connection_with_tunnel(
    dbname,
    host="localhost",
    auth_dbname=None,
    port=27017,
    ssh=False,
    user="hyperopt",
    pw=None,
):
    if not _has_mongo:
        raise ImportError("pymongo and gridfs are required for MongoTrials")

    tunnel = None
    if ssh:
        local_port = int(numpy.random.randint(27500, 28000))
        tunnel = subprocess.Popen(
            ["ssh", "-NTf", "-L", "%i:%s:%i" % (local_port, "127.0.0.1", port), host]
        )
        time.sleep(0.5)
        connection = pymongo.MongoClient(
            "127.0.0.1", local_port, document_class=SON, w=1, journal=True
        )
    else:
        connection = pymongo.MongoClient(
            host, port, document_class=SON, w=1, journal=True
        )
        if user:
            if pw is None:
                pw = read_pw()
            if user == "hyperopt" and auth_dbname is None:
                auth_dbname = "admin"
            try:
                connection[dbname].authenticate(user, pw, source=auth_dbname)
            except AttributeError:
                connection[auth_dbname or dbname].command(
                    "authenticate", user=user, pwd=pw
                )
    return connection, tunnel


def connection_from_string(s):
    protocol, user, pw, host, port, dbname, collection, authdb = parse_url(s)
    if protocol == "mongo":
        ssh = False
    elif protocol in ("mongo+ssh", "ssh+mongo"):
        ssh = True
    else:
        raise ValueError("unrecognized protocol for MongoJobs", protocol)
    connection, tunnel = connection_with_tunnel(
        dbname=dbname,
        host=host,
        auth_dbname=authdb,
        port=port,
        ssh=ssh,
        user=user,
        pw=pw,
    )
    return connection, tunnel, connection[dbname], connection[dbname][collection]


def _insert_one(collection, doc):
    result = collection.insert_one(doc)
    return getattr(result, "inserted_id", result)


def _replace_one(collection, query, doc, upsert=False):
    if hasattr(collection, "replace_one"):
        return collection.replace_one(query, doc, upsert=upsert)
    return collection.save(doc)


def _update_one(collection, query, update, upsert=False):
    if hasattr(collection, "update_one"):
        return collection.update_one(query, update, upsert=upsert)
    return collection.update(query, update, upsert=upsert, multi=False)


def _delete_many(collection, query):
    if hasattr(collection, "delete_many"):
        return collection.delete_many(query)
    return collection.remove(query)


class MongoJobs:
    def __init__(self, db, jobs, gfs=None, exp_key=None):
        self.db = db
        self.jobs = jobs
        self.gfs = gfs if gfs is not None else gridfs.GridFS(db)
        self.exp_key = exp_key

    @classmethod
    def new_from_uri(cls, uri, exp_key=None):
        connection, tunnel, db, jobs = connection_from_string(uri)
        obj = cls(db, jobs, gridfs.GridFS(db), exp_key=exp_key)
        obj.connection = connection
        obj.tunnel = tunnel
        return obj

    from_connection_str = new_from_uri

    def _query(self, query=None, exp_key=None):
        q = {} if query is None else copy.deepcopy(query)
        key = self.exp_key if exp_key is None else exp_key
        if key is not None:
            q["exp_key"] = key
        return q

    def _drivers(self):
        return self.db["drivers"]

    def _attachment_key(self, name):
        return "attachments.%s" % name

    def insert(self, job):
        job = SONify(copy.deepcopy(job))
        if "state" not in job:
            job["state"] = JOB_STATE_NEW
        if "owner" not in job:
            job["owner"] = None
        if "book_time" not in job:
            job["book_time"] = None
        if "refresh_time" not in job:
            job["refresh_time"] = coarse_utcnow()
        if "exp_key" not in job and self.exp_key is not None:
            job["exp_key"] = self.exp_key
        try:
            return _insert_one(self.jobs, job)
        except Exception as exc:
            raise OperationFailure(str(exc))

    def insert_many(self, jobs):
        return [self.insert(job) for job in jobs]

    def find(self, query=None, fields=None, exp_key=None, **kwargs):
        return self.jobs.find(self._query(query, exp_key), fields, **kwargs)

    def find_one(self, query=None, fields=None, exp_key=None, **kwargs):
        return self.jobs.find_one(self._query(query, exp_key), fields, **kwargs)

    def count(self, query=None, exp_key=None):
        cursor = self.find(query, exp_key=exp_key)
        try:
            return cursor.count()
        except Exception:
            return self.jobs.count_documents(self._query(query, exp_key))

    def update(self, job, fields, exp_key=None):
        if isinstance(job, dict):
            query = {"_id": job["_id"]}
        else:
            query = {"_id": job}
        if exp_key is not None:
            query["exp_key"] = exp_key
        try:
            return _update_one(self.jobs, query, {"$set": SONify(copy.deepcopy(fields))})
        except Exception as exc:
            raise OperationFailure(str(exc))

    def delete(self, job):
        query = {"_id": job["_id"] if isinstance(job, dict) else job}
        return _delete_many(self.jobs, query)

    def delete_all(self, exp_key=None):
        return _delete_many(self.jobs, self._query({}, exp_key))

    def reserve(self, owner=None, exp_key=None):
        if owner is None:
            owner = (socket.gethostname(), os.getpid())
        now = coarse_utcnow()
        query = self._query(
            {"state": JOB_STATE_NEW, "$or": [{"owner": None}, {"owner": {"$exists": False}}]},
            exp_key,
        )
        change = {
            "$set": {
                "state": JOB_STATE_RUNNING,
                "owner": owner,
                "book_time": now,
                "refresh_time": now,
            }
        }
        try:
            if hasattr(self.jobs, "find_one_and_update"):
                after = ReturnDocument.AFTER if ReturnDocument is not None else True
                return self.jobs.find_one_and_update(
                    query, change, sort=[("_id", 1)], return_document=after
                )
            return self.jobs.find_and_modify(query=query, update=change, new=True)
        except Exception as exc:
            raise OperationFailure(str(exc))

    def put_attachment(self, job, name, data, content_type=None):
        if isinstance(data, str):
            data = data.encode()
        kwargs = {}
        if content_type is not None:
            kwargs["content_type"] = content_type
        file_id = self.gfs.put(data, **kwargs)
        self.update(job, {self._attachment_key(name): file_id})
        return file_id

    def get_attachment(self, job, name):
        if not isinstance(job, dict):
            job = self.jobs.find_one({"_id": job})
        try:
            file_id = job["attachments"][name]
        except KeyError:
            raise KeyError(name)
        return self.gfs.get(file_id).read()

    def delete_attachment(self, job, name):
        if not isinstance(job, dict):
            job = self.jobs.find_one({"_id": job})
        try:
            file_id = job["attachments"][name]
        except KeyError:
            return
        try:
            self.gfs.delete(file_id)
        finally:
            _update_one(
                self.jobs,
                {"_id": job["_id"]},
                {"$unset": {self._attachment_key(name): 1}},
            )

    def attachment_names(self, job):
        if not isinstance(job, dict):
            job = self.jobs.find_one({"_id": job})
        return list(job.get("attachments", {}).keys())

    def get_driver(self, exp_key=None):
        key = self.exp_key if exp_key is None else exp_key
        return self._drivers().find_one({"exp_key": key})

    def set_driver(self, doc, exp_key=None):
        key = self.exp_key if exp_key is None else exp_key
        item = SONify(copy.deepcopy(doc))
        item["exp_key"] = key
        _replace_one(self._drivers(), {"exp_key": key}, item, upsert=True)
        return item

    def put_driver_attachment(self, name, data, exp_key=None):
        key = self.exp_key if exp_key is None else exp_key
        doc = self.get_driver(key)
        if doc is None:
            doc = {"exp_key": key}
            _insert_one(self._drivers(), doc)
        if isinstance(data, str):
            data = data.encode()
        file_id = self.gfs.put(data)
        _update_one(
            self._drivers(),
            {"exp_key": key},
            {"$set": {"attachments.%s" % name: file_id}},
        )
        return file_id

    def get_driver_attachment(self, name, exp_key=None):
        doc = self.get_driver(exp_key)
        if doc is None:
            raise KeyError(name)
        return self.gfs.get(doc["attachments"][name]).read()


class _MongoAttachments:
    def __init__(self, trials):
        self.trials = trials

    def __getitem__(self, name):
        return self.trials._mongo_jobs.get_driver_attachment(name)

    def __setitem__(self, name, value):
        self.trials._mongo_jobs.put_driver_attachment(name, value)

    def __delitem__(self, name):
        jobs = self.trials._mongo_jobs
        doc = jobs.get_driver()
        file_id = doc["attachments"][name]
        jobs.gfs.delete(file_id)
        _update_one(
            jobs._drivers(),
            {"_id": doc["_id"]},
            {"$unset": {"attachments.%s" % name: 1}},
        )

    def __contains__(self, name):
        doc = self.trials._mongo_jobs.get_driver()
        return doc is not None and name in doc.get("attachments", {})

    def keys(self):
        doc = self.trials._mongo_jobs.get_driver()
        return list((doc or {}).get("attachments", {}).keys())


class MongoTrials(Trials):
    def __init__(self, arg, exp_key=None, refresh=True):
        if isinstance(arg, MongoJobs):
            self._mongo_jobs = arg
            if exp_key is not None:
                self._mongo_jobs.exp_key = exp_key
        elif isinstance(arg, str):
            self._mongo_jobs = MongoJobs.new_from_uri(arg, exp_key=exp_key)
        else:
            raise TypeError("MongoTrials requires a MongoJobs instance or Mongo URI")
        self._exp_key = self._mongo_jobs.exp_key
        super().__init__(refresh=False)
        self.attachments = _MongoAttachments(self)
        if refresh:
            self.refresh()

    @property
    def exp_key(self):
        return self._exp_key

    @property
    def trials(self):
        return self._trials

    def refresh(self):
        self._trials = list(self._mongo_jobs.find())
        self._dynamic_trials = self._trials
        return self

    def insert_trial_docs(self, docs):
        docs = list(docs)
        for doc in docs:
            doc = SONify(copy.deepcopy(doc))
            if "_id" in doc:
                del doc["_id"]
            if "exp_key" not in doc:
                doc["exp_key"] = self._exp_key
            self._mongo_jobs.insert(doc)
        self.refresh()
        return docs

    def delete_all(self):
        self._mongo_jobs.delete_all()
        self.refresh()

    def trial_attachments(self, trial):
        mongo = self._mongo_jobs

        class Attachments:
            def __getitem__(self, name):
                return mongo.get_attachment(trial, name)

            def __setitem__(self, name, value):
                return mongo.put_attachment(trial, name, value)

            def __delitem__(self, name):
                return mongo.delete_attachment(trial, name)

            def __contains__(self, name):
                return name in mongo.attachment_names(trial)

            def keys(self):
                return mongo.attachment_names(trial)

        return Attachments()


class MongoCtrl(Ctrl):
    def __init__(self, trials, current_trial, read_only=False):
        self.trials = trials
        self.current_trial = current_trial
        self.read_only = read_only
        self._logs = {"info": [], "warn": [], "error": [], "debug": []}
        self._mongo_jobs = trials._mongo_jobs if isinstance(trials, MongoTrials) else trials

    def _log(self, level, *args):
        message = " ".join(str(x) for x in args)
        self._logs.setdefault(level, []).append(message)
        return message

    def info(self, *args):
        return self._log("info", *args)

    def warn(self, *args):
        return self._log("warn", *args)

    warning = warn

    def error(self, *args):
        return self._log("error", *args)

    def debug(self, *args):
        return self._log("debug", *args)

    @property
    def attachments(self):
        return self.trials.trial_attachments(self.current_trial)

    def checkpoint(self, result=None):
        if self.read_only:
            return
        fields = {"refresh_time": coarse_utcnow()}
        if result is not None:
            fields["result"] = SONify(copy.deepcopy(result))
        updates = {}
        for level, values in self._logs.items():
            if values:
                updates["logs.%s" % level] = {"$each": values}
        if updates:
            try:
                _update_one(
                    self._mongo_jobs.jobs,
                    {"_id": self.current_trial["_id"]},
                    {"$set": fields, "$push": updates},
                )
            except Exception as exc:
                raise OperationFailure(str(exc))
        else:
            self._mongo_jobs.update(self.current_trial, fields)
        self._logs = {"info": [], "warn": [], "error": [], "debug": []}
        if result is not None:
            self.current_trial["result"] = result
        return result

    def inject_results(self, specs, results, miscs, new_tids=None):
        if new_tids is None:
            existing = [t.get("tid", -1) for t in self.trials.trials]
            new_tids = list(range(max(existing + [-1]) + 1, max(existing + [-1]) + 1 + len(specs)))
        docs = []
        for tid, spec, result, misc in zip(new_tids, specs, results, miscs):
            misc = copy.deepcopy(misc)
            misc["tid"] = tid
            docs.append(
                {
                    "tid": tid,
                    "spec": spec,
                    "result": result,
                    "misc": misc,
                    "state": JOB_STATE_NEW,
                    "owner": None,
                    "book_time": None,
                    "refresh_time": coarse_utcnow(),
                    "exp_key": self.trials.exp_key,
                }
            )
        self.trials.insert_trial_docs(docs)
        return docs


def _evaluate_job(job, ctrl, mongo_jobs):
    cmd = job.get("cmd")
    if cmd is None:
        raise InvalidMongoTrial("job has no cmd")

    if isinstance(cmd, str):
        fn = json_call(cmd)
        return fn(job.get("spec"), ctrl)

    if cmd[0] == "call":
        fn = json_call(*cmd[1:])
        return fn(job.get("spec"), ctrl)

    if cmd[0] == "domain_attachment":
        name = cmd[1] if len(cmd) > 1 else "domain"
        payload = mongo_jobs.get_attachment(job, name)
        domain = pickler.loads(payload)
        return domain.evaluate(job.get("spec"), ctrl)

    if cmd[0] == "driver_attachment":
        name = cmd[1] if len(cmd) > 1 else "domain"
        domain = pickler.loads(mongo_jobs.get_driver_attachment(name))
        return domain.evaluate(job.get("spec"), ctrl)

    fn = json_call(*cmd)
    return fn(job.get("spec"), ctrl)


def main_worker(
    mongo_jobs,
    poll_interval=3.0,
    max_jobs=None,
    max_wait=None,
    reserve_timeout=None,
    workdir=None,
    exp_key=None,
):
    if not isinstance(mongo_jobs, MongoJobs):
        mongo_jobs = MongoJobs.new_from_uri(mongo_jobs, exp_key=exp_key)

    completed = 0
    started_waiting = time.time()
    while max_jobs is None or completed < max_jobs:
        job = mongo_jobs.reserve(exp_key=exp_key)
        if job is None:
            elapsed = time.time() - started_waiting
            if reserve_timeout is not None and elapsed >= reserve_timeout:
                raise ReserveTimeout()
            if max_wait is not None and elapsed >= max_wait:
                raise WaitQuit()
            time.sleep(poll_interval)
            continue

        started_waiting = time.time()
        trials = MongoTrials(mongo_jobs, exp_key=job.get("exp_key"), refresh=False)
        ctrl = MongoCtrl(trials, job)
        try:
            directory = workdir
            if directory is None:
                driver = mongo_jobs.get_driver(job.get("exp_key"))
                if driver is not None:
                    directory = driver.get("workdir")
            if directory:
                with working_dir(directory):
                    result = _evaluate_job(job, ctrl, mongo_jobs)
            else:
                result = _evaluate_job(job, ctrl, mongo_jobs)
            ctrl.checkpoint(result)
            mongo_jobs.update(
                job,
                {
                    "state": JOB_STATE_DONE,
                    "result": SONify(result),
                    "refresh_time": coarse_utcnow(),
                },
            )
        except (KeyboardInterrupt, SystemExit, Shutdown):
            mongo_jobs.update(
                job,
                {
                    "state": JOB_STATE_ERROR,
                    "error": "worker interrupted",
                    "refresh_time": coarse_utcnow(),
                },
            )
            raise
        except Exception as exc:
            logger.exception("job exception")
            try:
                ctrl.error(repr(exc))
                ctrl.checkpoint()
            except Exception:
                pass
            mongo_jobs.update(
                job,
                {
                    "state": JOB_STATE_ERROR,
                    "error": repr(exc),
                    "refresh_time": coarse_utcnow(),
                },
            )
        completed += 1
    return completed


def worker_launch(
    mongo,
    poll_interval=3.0,
    max_jobs=None,
    max_wait=None,
    reserve_timeout=None,
    workdir=None,
    exp_key=None,
):
    return main_worker(
        mongo,
        poll_interval=poll_interval,
        max_jobs=max_jobs,
        max_wait=max_wait,
        reserve_timeout=reserve_timeout,
        workdir=workdir,
        exp_key=exp_key,
    )


def mongo_worker(argv=None):
    parser = optparse.OptionParser()
    parser.add_option("--loop", action="store_true", default=False)
    parser.add_option("--poll-interval", type="float", default=3.0)
    parser.add_option("--max-jobs", type="int", default=None)
    parser.add_option("--max-wait", type="float", default=None)
    parser.add_option("--reserve-timeout", type="float", default=None)
    parser.add_option("--workdir", default=None)
    parser.add_option("--exp-key", default=None)
    options, args = parser.parse_args(argv)
    if len(args) != 1:
        parser.error("expected a mongo URI")
    return main_worker(
        args[0],
        poll_interval=options.poll_interval,
        max_jobs=options.max_jobs if options.loop else 1,
        max_wait=options.max_wait,
        reserve_timeout=options.reserve_timeout,
        workdir=options.workdir,
        exp_key=options.exp_key,
    )


def main(argv=None):
    return mongo_worker(argv)


if __name__ == "__main__":
    main()
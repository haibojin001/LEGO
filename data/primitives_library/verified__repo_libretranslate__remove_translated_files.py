import atexit
import os
import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler


def remove_translated_files(upload_dir: str):
    now = time.mktime(datetime.now().timetuple())

    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        if os.path.isfile(filepath):
            modified_time = os.path.getmtime(filepath)
            if now - modified_time > 1800:
                os.remove(filepath)


def setup(upload_dir):
    scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
    scheduler.add_job(
        remove_translated_files,
        "interval",
        minutes=30,
        kwargs={"upload_dir": upload_dir},
    )
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
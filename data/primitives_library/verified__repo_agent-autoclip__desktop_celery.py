import os
from pathlib import Path

if os.getenv("AUTOCLIP_DESKTOP_MODE", "").casefold() not in ("1", "true", "yes"):
    raise RuntimeError("此模块仅在桌面模式下可用")

from celery import Celery

app_dir = Path(
    os.getenv("AUTOCLIP_APP_DIR", "~/Library/Application Support/AutoClip")
).expanduser()
app_dir.mkdir(parents=True, exist_ok=True)

celery_dir = app_dir / "celery"
celery_dir.mkdir(parents=True, exist_ok=True)

(celery_dir / "in").mkdir(exist_ok=True)
(celery_dir / "out").mkdir(exist_ok=True)
(celery_dir / "processed").mkdir(exist_ok=True)

celery_app = Celery(
    "autoclip_desktop",
    broker="filesystem://",
    backend=f"db+sqlite:///{celery_dir / 'results.sqlite3'}",
)

celery_app.conf.update(
    broker_transport_options={
        "data_folder_in": str(celery_dir / "in"),
        "data_folder_out": str(celery_dir / "out"),
        "data_folder_processed": str(celery_dir / "processed"),
    },
    task_ignore_result=False,
)

if __name__ == "__main__":
    celery_app.start()
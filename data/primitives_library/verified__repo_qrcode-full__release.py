"""Release hook helpers for qrcode."""

import datetime
import re
from pathlib import Path


def update_manpage(data):
    """Update the qrcode version and date in the manpage."""
    if data["name"] != "qrcode":
        return

    root = Path(__file__).parent.parent.resolve()
    manpage = root / "doc" / "qr.1"

    with manpage.open("r") as stream:
        contents = stream.readlines()

    modified = False
    for index, entry in enumerate(contents):
        if not entry.startswith(".TH "):
            continue

        fields = re.split(r'"([^"]*)"', entry)
        if len(fields) < 5:
            continue

        modified = fields[3] != data["new_version"]
        if modified:
            fields[3] = data["new_version"]
            fields[1] = datetime.datetime.now(
                tz=datetime.timezone.utc
            ).strftime("%-d %b %Y")
            contents[index] = '"'.join(fields)
        break

    if modified:
        with manpage.open("w") as stream:
            for entry in contents:
                stream.write(entry)
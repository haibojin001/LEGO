import datetime
import re
from pathlib import Path


def update_manpage(data):
    if data["name"] != "qrcode":
        return

    manpage = Path(__file__).resolve().parents[1] / "doc" / "qr.1"

    with manpage.open("r") as source:
        contents = source.readlines()

    modified = False

    for index, text in enumerate(contents):
        if not text.startswith(".TH "):
            continue

        fields = re.split(r'"([^"]*)"', text)
        if len(fields) < 5:
            continue

        modified = fields[3] != data["new_version"]

        if modified:
            fields[1] = datetime.datetime.now(
                tz=datetime.timezone.utc
            ).strftime("%-d %b %Y")
            fields[3] = data["new_version"]
            contents[index] = '"'.join(fields)

        break

    if modified:
        with manpage.open("w") as destination:
            destination.writelines(contents)
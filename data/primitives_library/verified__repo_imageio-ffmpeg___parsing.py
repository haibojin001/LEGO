import re
import threading
import time

from ._utils import logger


class LogCatcher(threading.Thread):
    """Continuously consume ffmpeg stderr without allowing its pipe to fill."""

    def __init__(self, file):
        self._file = file
        self._header = ""
        self._lines = []
        self._remainder = b""
        self._should_stop = False
        threading.Thread.__init__(self)
        self.daemon = True
        self.start()

    def stop_me(self):
        self._should_stop = True

    @property
    def header(self):
        """Return parsed ffmpeg header text, if available."""
        return self._header

    def get_text(self, timeout=0):
        """Return stderr text collected so far."""
        if timeout > 0:
            deadline = time.time() + timeout
            while self.is_alive() and time.time() < deadline:  # pragma: no cover
                time.sleep(0.01)

        body = b"\n".join(self._lines).decode("utf-8", "ignore")
        return self._header + "\n" + body

    def run(self):
        trim_lines = limit_lines

        while not self._should_stop:
            time.sleep(0)

            try:
                chunk = self._file.read(20)
            except ValueError:  # pragma: no cover
                break

            if not chunk:
                break

            chunk = chunk.replace(b"\r", b"\n").replace(b"\n\n", b"\n")
            pieces = chunk.split(b"\n")
            pieces[0] = self._remainder + pieces[0]
            self._remainder = pieces.pop()

            self._lines.extend(pieces)

            if not self._header:
                if get_output_video_line(self._lines):
                    self._header += b"\n".join(self._lines).decode(
                        "utf-8", "ignore"
                    )
            elif self._lines:
                self._lines = trim_lines(self._lines)

        try:
            self._file.close()
        except Exception:
            pass


def get_output_video_line(lines):
    """Return the video stream description found in ffmpeg output."""
    output_seen = False

    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith(b"Output "):
            output_seen = True
        elif output_seen and stripped.startswith(b"Stream ") and b" Video:" in stripped:
            return line


def limit_lines(lines, N=32):
    """Keep only a trailing subset once the line collection becomes large."""
    if len(lines) > N * 2:
        return [b"... showing only last few lines ..."] + lines[-N:]
    return lines


def cvsecs(*args):
    """Convert seconds, minutes/seconds, or hours/minutes/seconds to seconds."""
    count = len(args)

    if count == 1:
        return float(args[0])
    if count == 2:
        return float(args[0]) * 60 + float(args[1])
    if count == 3:
        return float(args[0]) * 3600 + float(args[1]) * 60 + float(args[2])


def parse_ffmpeg_header(text):
    lines = text.splitlines()
    meta = {}

    version_text = lines[0].split("version", 1)[-1].split("Copyright")[0]
    meta["ffmpeg_version"] = version_text.strip() + " " + lines[1].strip()

    video_lines = [
        line
        for line in lines
        if line.lstrip().startswith("Stream ") and " Video: " in line
    ]

    first_video = video_lines[0]
    video_description = first_video.split("Video: ", 1)[-1]

    meta["codec"] = video_description.lstrip().split(" ", 1)[0].strip()
    meta["pix_fmt"] = re.split(
        r",\s*(?![^()]*\))",
        video_description,
    )[1].strip()

    audio_lines = [
        line
        for line in lines
        if line.lstrip().startswith("Stream ") and " Audio: " in line
    ]

    if audio_lines:
        audio_description = audio_lines[0].split("Audio: ", 1)[-1]
        meta["audio_codec"] = audio_description.lstrip().split(" ", 1)[0].strip()

    fps = 0
    fps_matches = re.findall(r" ([0-9]+\.?[0-9]*) (fps)", first_video)
    if fps_matches:
        fps = float(fps_matches[0][0].strip())
    meta["fps"] = fps

    source_match = re.search(" [0-9]*x[0-9]*(,| )", first_video)
    source_parts = first_video[
        source_match.start() : source_match.end() - 1
    ].split("x")
    meta["source_size"] = tuple(map(int, source_parts))

    output_video = video_lines[-1]
    output_match = re.search(" [0-9]*x[0-9]*(,| )", output_video)
    output_parts = output_video[
        output_match.start() : output_match.end() - 1
    ].split("x")
    meta["size"] = tuple(map(int, output_parts))

    if meta["source_size"] != meta["size"]:
        logger.warning(
            "The frame size for reading {} is different from the source "
            "frame size {}.".format(meta["size"], meta["source_size"])
        )

    rotation_match = re.compile(r"rotate\s+:\s([0-9]+)").search(text)
    meta["rotate"] = int(rotation_match.groups()[0]) if rotation_match else 0

    duration_line = [line for line in lines if "Duration: " in line][0]
    duration_match = re.search(
        " [0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9]",
        duration_line,
    )

    duration = 0
    if duration_match is not None:
        time_parts = duration_line[
            duration_match.start() + 1 : duration_match.end()
        ].split(":")
        duration = cvsecs(*time_parts)

    meta["duration"] = duration
    return meta
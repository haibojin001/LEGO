import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache

from ._parsing import LogCatcher, cvsecs, parse_ffmpeg_header
from ._utils import _popen_kwargs, get_ffmpeg_exe, logger


ISWIN = sys.platform.startswith("win")


h264_encoder_preference = defaultdict(lambda: -1)
h264_encoder_preference["libx264"] = 100
h264_encoder_preference["h264_nvenc"] = 90
h264_encoder_preference["nvenc_h264"] = 90
h264_encoder_preference["nvenc"] = 90
h264_encoder_preference["h264_vaapi"] = 80
h264_encoder_preference["libopenh264"] = 70
h264_encoder_preference["libx264rgb"] = 50


def ffmpeg_test_encoder(encoder):
    command = [
        get_ffmpeg_exe(),
        "-hide_banner",
        "-f",
        "lavfi",
        "-i",
        "nullsrc=s=256x256:d=8",
        "-vcodec",
        encoder,
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def get_compiled_h264_encoders():
    command = [get_ffmpeg_exe(), "-hide_banner", "-encoders"]
    result = subprocess.run(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    text = result.stdout.decode().replace("\r", "")
    parts = text.split("------")
    footer = parts[1].strip("\n")

    encoders = []
    for row in footer.split("\n"):
        row = row.strip()
        encoder = row.split(" ")[1]

        if encoder in h264_encoder_preference:
            encoders.append(encoder)
        elif row[0] == "V" and "H.264" in row:
            encoders.append(encoder)

    encoders.sort(key=lambda name: h264_encoder_preference[name], reverse=True)

    if "h264_nvenc" in encoders:
        for deprecated_name in ("nvenc", "nvenc_h264"):
            if deprecated_name in encoders:
                encoders.remove(deprecated_name)

    return tuple(encoders)


@lru_cache()
def get_first_available_h264_encoder():
    for encoder in get_compiled_h264_encoders():
        if ffmpeg_test_encoder(encoder):
            return encoder

    raise RuntimeError(
        "No valid H.264 encoder was found with the ffmpeg installation"
    )


def count_frames_and_secs(path):
    if isinstance(path, pathlib.PurePath):
        path = str(path)
    if not isinstance(path, str):
        raise TypeError("Video path must be a string or pathlib.Path.")

    command = [
        get_ffmpeg_exe(),
        "-i",
        path,
        "-map",
        "0:v:0",
        "-vf",
        "null",
        "-f",
        "null",
        "-",
    ]

    try:
        output = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            **_popen_kwargs()
        )
    except subprocess.CalledProcessError as err:
        text = err.output.decode(errors="ignore")
        raise RuntimeError(
            "FFMPEG call failed with {}:\n{}".format(err.returncode, text)
        )

    for row in reversed(output.splitlines()):
        if row.startswith(b"frame="):
            row_text = row.decode(errors="ignore")

            position = row_text.find("frame=")
            frames = None
            seconds = None

            if position >= 0:
                value = row_text[position:].split("=", 1)[1]
                value = value.lstrip().split(" ", 1)[0].strip()
                frames = int(value)

            position = row_text.find("time=")
            if position >= 0:
                value = row_text[position:].split("=", 1)[1]
                value = value.lstrip().split(" ", 1)[0].strip()
                seconds = cvsecs(*value.split(":"))

            return frames, seconds

    raise RuntimeError("Could not get number of frames")


def read_frames(
    path,
    pix_fmt="rgb24",
    bpp=None,
    input_params=None,
    output_params=None,
    bits_per_pixel=None,
):
    if isinstance(path, pathlib.PurePath):
        path = str(path)
    if not isinstance(path, str):
        raise TypeError("Video path must be a string or pathlib.Path.")

    if input_params is None:
        input_params = []
    if output_params is None:
        output_params = []

    if bpp is not None:
        logger.warning(
            "The bpp argument is deprecated and will be removed in a future "
            "version. Use bits_per_pixel instead."
        )
        if bits_per_pixel is None:
            bits_per_pixel = bpp * 8

    if bits_per_pixel is None:
        bits_per_pixel = 24

    command = [get_ffmpeg_exe()]
    command.extend(input_params)
    command.extend(
        [
            "-i",
            path,
            "-f",
            "image2pipe",
            "-pix_fmt",
            pix_fmt,
            "-vcodec",
            "rawvideo",
        ]
    )
    command.extend(output_params)
    command.append("-")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_popen_kwargs()
    )
    catcher = LogCatcher(process.stderr)

    try:
        while catcher.header is None:
            if process.poll() is not None:
                message = catcher.get_text()
                raise IOError(
                    "Could not load meta information\n=== stderr ===\n{}".format(
                        message
                    )
                )
            time.sleep(0.01)

        metadata = parse_ffmpeg_header(catcher.header)
        yield metadata

        width, height = metadata["size"]
        frame_size = int(width * height * bits_per_pixel / 8)

        while True:
            data = b""

            while len(data) < frame_size:
                chunk = process.stdout.read(frame_size - len(data))
                if not chunk:
                    if not data:
                        return
                    raise RuntimeError(
                        "End of file reached before full frame could be read."
                    )
                data += chunk

            yield data
    finally:
        try:
            if process.stdout is not None:
                process.stdout.close()
        except Exception:
            pass

        try:
            if process.stderr is not None:
                process.stderr.close()
        except Exception:
            pass

        if process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass

        try:
            process.wait()
        except Exception:
            pass


def write_frames(
    path,
    size,
    pix_fmt_in="rgb24",
    pix_fmt_out="yuv420p",
    fps=16,
    quality=5,
    bitrate=None,
    codec=None,
    macro_block_size=16,
    ffmpeg_log_level="warning",
    ffmpeg_timeout=None,
    input_params=None,
    output_params=None,
    audio_path=None,
    audio_codec=None,
):
    if isinstance(path, pathlib.PurePath):
        path = str(path)
    if not isinstance(path, str):
        raise TypeError("Video path must be a string or pathlib.Path.")

    if not isinstance(size, tuple) or len(size) != 2:
        raise TypeError("size must be a tuple of two integers.")

    if not isinstance(size[0], int) or not isinstance(size[1], int):
        raise TypeError("size must be a tuple of two integers.")

    if input_params is None:
        input_params = []
    if output_params is None:
        output_params = []

    if macro_block_size is None:
        macro_block_size = 1

    if macro_block_size > 1:
        width, height = size
        new_width = (
            int((width + macro_block_size - 1) / macro_block_size)
            * macro_block_size
        )
        new_height = (
            int((height + macro_block_size - 1) / macro_block_size)
            * macro_block_size
        )

        if new_width != width or new_height != height:
            logger.warning(
                "IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible "
                "by macro_block_size={}, resizing from {} to {} to ensure video "
                "compatibility with most codecs and players. To prevent resizing, "
                "make your input divisible by {} or set macro_block_size to 1 "
                "(may result in a video that is not compatible with some players).".format(
                    macro_block_size,
                    (width, height),
                    (new_width, new_height),
                    macro_block_size,
                )
            )
            size = (new_width, new_height)

    if codec is None:
        codec = get_first_available_h264_encoder()

    command = [
        get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "{}x{}".format(size[0], size[1]),
        "-pix_fmt",
        pix_fmt_in,
        "-r",
        "{:0.02f}".format(fps),
    ]
    command.extend(input_params)
    command.extend(["-i", "-"])

    if audio_path is not None:
        if isinstance(audio_path, pathlib.PurePath):
            audio_path = str(audio_path)
        command.extend(["-i", audio_path])

    command.extend(["-vcodec", codec])

    if bitrate is not None:
        command.extend(["-b:v", str(bitrate)])
    elif quality is not None:
        crf = int((1.0 - float(quality) / 10.0) * 51.0)
        command.extend(["-crf", str(crf)])

    command.extend(["-pix_fmt", pix_fmt_out])

    if audio_codec is not None:
        command.extend(["-acodec", audio_codec])

    command.extend(output_params)
    command.extend(["-loglevel", ffmpeg_log_level, path])

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_popen_kwargs()
    )
    catcher = LogCatcher(process.stderr)

    try:
        while True:
            frame = yield

            if process.poll() is not None:
                raise IOError(
                    "FFMPEG subprocess terminated unexpectedly:\n{}".format(
                        catcher.get_text()
                    )
                )

            try:
                process.stdin.write(frame)
            except Exception as err:
                raise IOError(
                    "FFMPEG subprocess could not write frame:\n{}".format(
                        catcher.get_text()
                    )
                ) from err
    finally:
        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:
            pass

        try:
            if ffmpeg_timeout is None:
                process.wait()
            else:
                process.wait(timeout=ffmpeg_timeout)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            finally:
                process.wait()
        except Exception:
            pass

        try:
            if process.stdout is not None:
                process.stdout.close()
        except Exception:
            pass

        try:
            if process.stderr is not None:
                process.stderr.close()
        except Exception:
            pass

        if process.returncode:
            logger.warning(
                "IMAGEIO FFMPEG_WRITER WARNING: ffmpeg exited with code {}:\n{}".format(
                    process.returncode,
                    catcher.get_text(),
                )
            )
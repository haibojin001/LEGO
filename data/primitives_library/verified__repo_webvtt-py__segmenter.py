"""Utilities for splitting WebVTT files into HLS-compatible segments."""

import os
import pathlib
import typing
from math import ceil, floor

from .webvtt import Caption, WebVTT


DEFAULT_MPEGTS = 900000
DEFAULT_SECONDS = 10


def segment(
    webvtt_path: str,
    output: str,
    seconds: int = DEFAULT_SECONDS,
    mpegts: int = DEFAULT_MPEGTS
):
    """Create segmented WebVTT files and an HLS playlist."""
    captions = WebVTT.read(webvtt_path).captions
    output_folder = pathlib.Path(output)

    os.makedirs(output_folder, exist_ok=True)

    segments = slice_segments(captions, seconds)
    write_segments(output_folder, segments, mpegts)
    write_manifest(output_folder, segments, seconds)


def slice_segments(
    captions: typing.Sequence[Caption],
    seconds: int
) -> typing.List[typing.List[Caption]]:
    """Split captions into groups corresponding to fixed-duration segments."""
    count = 0
    if captions:
        count = int(ceil(captions[-1].end_in_seconds / seconds))

    result: typing.List[typing.List[Caption]] = [[] for _ in range(count)]

    for caption in captions:
        first_segment = floor(caption.start_in_seconds / seconds)
        result[first_segment].append(caption)

        last_segment = floor(caption.end_in_seconds / seconds)
        if last_segment > first_segment:
            for segment_number in range(first_segment + 1, last_segment + 1):
                result[segment_number].append(caption)

    return result


def write_segments(
    output_folder: pathlib.Path,
    segments: typing.Iterable[typing.Iterable[Caption]],
    mpegts: int
):
    """Write caption segment files into an output directory."""
    for number, captions in enumerate(segments):
        filename = output_folder / f"fileSequence{number}.webvtt"

        with open(filename, "w", encoding="utf-8") as file:
            file.write("WEBVTT\n")
            file.write(
                f"X-TIMESTAMP-MAP=MPEGTS:{mpegts},LOCAL:00:00:00.000\n"
            )

            for caption in captions:
                file.write(f"\n{caption.start} --> {caption.end}\n")
                file.writelines(f"{line}\n" for line in caption.lines)


def write_manifest(
    output_folder: pathlib.Path,
    segments: typing.Iterable[typing.Iterable[Caption]],
    seconds: int
):
    """Write the HLS playlist describing the generated segments."""
    filename = output_folder / "prog_index.m3u8"

    with open(filename, "w", encoding="utf-8") as file:
        file.write("#EXTM3U\n")
        file.write(f"#EXT-X-TARGETDURATION:{seconds}\n")
        file.write("#EXT-X-VERSION:3\n")
        file.write("#EXT-X-PLAYLIST-TYPE:VOD\n")

        for number, _ in enumerate(segments):
            file.write("#EXTINF:30.00000\n")
            file.write(f"fileSequence{number}.webvtt\n")

        file.write("#EXT-X-ENDLIST\n")
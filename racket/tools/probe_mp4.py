"""不依赖 ffmpeg/opencv 的 MP4 探测：直接读 ISO BMFF 盒子。

用途：在没装解码器时也能知道视频的分辨率、帧率、时长和帧数，
方便先规划抽帧策略。只读文件头，不会把 2 GB 视频读进内存。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterator

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}


def _iter_boxes(fh, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        header = fh.read(8)
        if len(header) < 8:
            return
        size, kind = struct.unpack(">I4s", header)
        offset = 8
        if size == 1:
            raw = fh.read(8)
            if len(raw) < 8:
                return
            size = struct.unpack(">Q", raw)[0]
            offset = 16
        elif size == 0:
            size = end - pos
        if size < offset:
            return
        yield kind, pos + offset, pos + size
        pos += size


def _find(fh, start: int, end: int, path: list[bytes]) -> tuple[int, int] | None:
    for kind, body, box_end in _iter_boxes(fh, start, end):
        if kind == path[0]:
            if len(path) == 1:
                return body, box_end
            if kind in _CONTAINERS:
                found = _find(fh, body, box_end, path[1:])
                if found:
                    return found
    return None


def _read(fh, start: int, count: int) -> bytes:
    fh.seek(start)
    return fh.read(count)


def _timescale_duration(fh, body: int, version: int) -> tuple[int, float]:
    if version == 1:
        raw = _read(fh, body + 4, 8 + 8 + 4 + 8)
        _, _, scale, duration = struct.unpack(">QQIQ", raw)
    else:
        raw = _read(fh, body + 4, 4 + 4 + 4 + 4)
        _, _, scale, duration = struct.unpack(">IIII", raw)
    return scale, float(duration)


def probe_mp4(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    size = path.stat().st_size
    info: dict[str, Any] = {"path": str(path), "size_mb": round(size / 1048576, 1)}

    with path.open("rb") as fh:
        kind, body, box_end = next(iter(_iter_boxes(fh, 0, size)), (None, 0, 0))
        info["container"] = kind.decode("latin1", "replace") if kind else None

        moov = _find(fh, 0, size, [b"moov"])
        if not moov:
            info["error"] = "找不到 moov（可能还没写完，或不是 MP4）"
            return info
        moov_body, moov_end = moov

        mvhd = _find(fh, moov_body, moov_end, [b"mvhd"])
        if mvhd:
            version = _read(fh, mvhd[0], 1)[0]
            scale, duration = _timescale_duration(fh, mvhd[0], version)
            if scale:
                info["duration_s"] = round(duration / scale, 3)

        tracks: list[dict[str, Any]] = []
        for tk_kind, tk_body, tk_end in _iter_boxes(fh, moov_body, moov_end):
            if tk_kind != b"trak":
                continue
            track: dict[str, Any] = {}
            tkhd = _find(fh, tk_body, tk_end, [b"tkhd"])
            if tkhd:
                version = _read(fh, tkhd[0], 1)[0]
                base = tkhd[0] + (4 + 8 + 8 + 4 + 4 + 8 if version == 1 else 4 + 4 + 4 + 4 + 4 + 4)
                wh = _read(fh, base + 8 + 2 + 2 + 2 + 2 + 36, 8)
                if len(wh) == 8:
                    width, height = struct.unpack(">II", wh)
                    track["width"] = round(width / 65536)
                    track["height"] = round(height / 65536)
            mdhd = _find(fh, tk_body, tk_end, [b"mdhd"])
            if mdhd:
                version = _read(fh, mdhd[0], 1)[0]
                scale, duration = _timescale_duration(fh, mdhd[0], version)
                track["media_timescale"] = scale
                track["media_duration_s"] = round(duration / scale, 3) if scale else None
            stts = _find(fh, tk_body, tk_end, [b"mdia", b"minf", b"stbl", b"stts"])
            if stts:
                body_size = stts[1] - stts[0]
                raw = _read(fh, stts[0], min(body_size, 8))
                if len(raw) == 8:
                    entry_count = struct.unpack(">I", raw[4:8])[0]
                    entries = _read(fh, stts[0] + 8, entry_count * 8)
                    total = 0
                    for i in range(entry_count):
                        count, _delta = struct.unpack(">II", entries[i * 8:(i + 1) * 8])
                        total += count
                    track["samples"] = total
                    if track.get("media_duration_s"):
                        track["fps"] = round(total / track["media_duration_s"], 3)
            tracks.append(track)

        video = next((t for t in tracks if t.get("width")), None)
        if video:
            info.update({
                "width": video.get("width"),
                "height": video.get("height"),
                "fps": video.get("fps"),
                "frames": video.get("samples"),
            })
        # mdhd 缺失时用总帧数/总时长反推帧率
        if info.get("fps") is None and info.get("frames") and info.get("duration_s"):
            info["fps"] = round(info["frames"] / info["duration_s"], 3)
            info["fps_source"] = "frames/duration"
        if info.get("fps"):
            # 抽到约 4 fps 需要的步长，用来把帧数压到止损检查点的量级
            info["stride_for_4fps"] = max(1, round(info["fps"] / 4.0))
            info["frames_at_4fps"] = int(round(info["duration_s"] * 4))
        info["tracks"] = len(tracks)
    return info


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    for target in sys.argv[1:]:
        print(json.dumps(probe_mp4(target), ensure_ascii=False, indent=2))

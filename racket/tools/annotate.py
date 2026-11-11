"""本地标注/复核服务：浏览器里改框，直接写回 JSONL。

    python -m racket.tools.annotate --labels data/labels.jsonl --port 8710

启动后打开 http://127.0.0.1:8710 。页面只读写这一个标注文件，
图片按帧序号提供，不接受任意路径，避免路径穿越。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

from ..dataset import BoxLabel, DatasetError, FrameLabel, load_frames, save_frames

PAGE = Path(__file__).with_name("annotator.html")


class LabelStore:
    """标注文件的内存副本 + 原子写回。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames = load_frames(path)
        self._lock = threading.Lock()

    def as_json(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{
                "index": i,
                "session_id": f.session_id,
                "frame_id": f.frame_id,
                "timestamp_s": f.timestamp_s,
                "image_size": list(f.image_size),
                "reviewed": f.reviewed,
                "boxes": [b.to_dict() for b in f.boxes],
            } for i, f in enumerate(self.frames)]

    def image_path(self, index: int) -> Path:
        with self._lock:
            return Path(self.frames[index].image_path)

    def update(self, index: int, boxes: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            frame = self.frames[index]
            parsed: list[BoxLabel] = []
            for raw in boxes:
                bbox = raw.get("bbox_xyxy")
                box = BoxLabel(
                    bbox_xyxy=tuple(float(v) for v in bbox) if bbox else None,
                    side=raw.get("side", "UNKNOWN"),
                    visible=bool(raw.get("visible", True)),
                    hard=tuple(raw.get("hard", ())),
                )
                if box.visible:
                    # 复用契约校验：出界、退化框一律拒绝
                    from ..schema import check_bbox
                    box.bbox_xyxy = check_bbox(box.bbox_xyxy, frame.image_size)
                parsed.append(box)
            updated = FrameLabel(
                session_id=frame.session_id,
                frame_id=frame.frame_id,
                timestamp_s=frame.timestamp_s,
                image_path=frame.image_path,
                boxes=parsed,
                image_size=frame.image_size,
                # 保存即视为"已复核"：空框且 reviewed=True 才是真正的负样本
                reviewed=True,
            )
            self.frames[index] = updated
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            save_frames(self.frames, tmp)
            tmp.replace(self.path)
            return {"ok": True, "index": index, "boxes": len(parsed), "reviewed": True}


def build_handler(store: LabelStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RacketAnnotator/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # 安静一点
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: Any) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/frames":
                self._json(200, {"frames": store.as_json()})
                return
            if parsed.path == "/api/image":
                query = parse_qs(parsed.query)
                try:
                    index = int(query.get("i", ["-1"])[0])
                    path = store.image_path(index)
                except (ValueError, IndexError, KeyError):
                    self._json(404, {"error": "bad index"})
                    return
                if not path.exists():
                    self._json(404, {"error": f"图片不存在：{path.name}"})
                    return
                suffix = path.suffix.lower()
                mime = "image/png" if suffix == ".png" else "image/jpeg"
                self._send(200, path.read_bytes(), mime)
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = store.update(int(payload["index"]), payload.get("boxes", []))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, DatasetError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, result)

    return Handler


def serve(labels: str | Path, *, host: str = "127.0.0.1", port: int = 8710) -> None:
    store = LabelStore(Path(labels))
    httpd = ThreadingHTTPServer((host, port), build_handler(store))
    print(f"标注服务已启动：http://{host}:{port}  （标注文件 {labels}，共 {len(store.frames)} 帧）")
    print("按 Ctrl+C 退出；每次翻页会自动保存当前帧。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        httpd.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="球拍标注复核工具")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8710)
    args = parser.parse_args(argv)
    serve(args.labels, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

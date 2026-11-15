"""Write the raw camera frames of a session to an mp4 in a background thread; frame i of the file
is the i-th frame passed to write() (hand_frames.frame_idx)."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, path: Path, fps: float = 30.0, codec: str = "mp4v"):
        self.path, self.fps, self.codec = Path(path), float(fps), codec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.size: tuple[int, int] | None = None
        self._q: queue.Queue = queue.Queue(maxsize=120)
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def write(self, frame_bgr: np.ndarray) -> int:
        """Queue a copy of the frame; returns its index in the file."""
        if self.size is None:
            self.size = (frame_bgr.shape[1], frame_bgr.shape[0])
        assert (frame_bgr.shape[1], frame_bgr.shape[0]) == self.size, "frame size changed"
        self._q.put(frame_bgr.copy())
        self.n += 1
        return self.n - 1

    def _run(self) -> None:
        vw = None
        while (f := self._q.get()) is not None:
            if vw is None:
                vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*self.codec),
                                     self.fps, (f.shape[1], f.shape[0]))  # fmt: skip
            vw.write(f)
        if vw is not None:
            vw.release()

    def close(self) -> int:
        """Finish writing; returns the number of frames."""
        self._q.put(None)
        self._th.join()
        return self.n

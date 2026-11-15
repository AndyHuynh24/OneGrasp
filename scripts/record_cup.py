"""Record a cup-rig session: webcam video + the microcontroller's sensor log, for ground truth.

    python scripts/record_cup.py --port /dev/cu.usbmodem1101 --out data/own/cup/<participant>_<take>

Writes <out>/video.mp4, <out>/frames.csv (frame index, host time s) and <out>/sensor.log
(the raw serial lines, each prefixed with the host receive time). The LED must be visible in
the camera image. Then: python scripts/align_cup.py <out>. Press q to stop.
Needs pyserial (pip install pyserial). Participant split: decide val / test participants
before recording (docs/CUP_RIG.md).
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2


def serial_reader(port: str, baud: int, path: Path, stop: threading.Event) -> None:
    import serial  # pyserial, only needed here

    with serial.Serial(port, baud, timeout=0.1) as ser, open(path, "w") as f:
        ser.reset_input_buffer()
        while not stop.is_set():
            line = ser.readline()
            if line:
                f.write(f"{time.perf_counter():.6f} {line.decode(errors='replace').strip()}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial port of the board (ls /dev/cu.*)")
    ap.add_argument("--baud", type=int, default=230400)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)  # never overwrite a session
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    assert ok, "cannot read the camera"
    h, w = frame.shape[:2]
    wr = cv2.VideoWriter(
        str(args.out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    stop = threading.Event()
    th = threading.Thread(
        target=serial_reader, args=(args.port, args.baud, args.out / "sensor.log", stop)
    )
    th.start()
    times = []
    try:
        while True:
            ok, frame = cap.read()
            t = time.perf_counter()
            if not ok:
                break
            wr.write(frame)
            times.append(t)
            cv2.imshow("recording (q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop.set()
        th.join()
        wr.release()
        cap.release()
        cv2.destroyAllWindows()
        with open(args.out / "frames.csv", "w") as f:
            f.write("frame,host_t\n")
            f.writelines(f"{i},{t:.6f}\n" for i, t in enumerate(times))
    dur = times[-1] - times[0] if len(times) > 1 else 0.0
    print(f"{len(times)} frames, {dur:.1f} s ({len(times) / max(dur, 1e-9):.1f} fps) -> {args.out}")


if __name__ == "__main__":
    main()

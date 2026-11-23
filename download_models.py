# -*- coding: utf-8 -*-
"""Download and verify the MediaPipe model files used by this project.

The binary model assets are intentionally not stored in Git. This script keeps
the checkout small and makes first-time setup reproducible: every download is
written to a temporary file, checked against a pinned SHA-256, and only then
atomically moved into ``models/``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"

MODELS = {
    "hand_landmarker.task": (
        f"{MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        7_819_105,
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "gesture_recognizer.task": (
        f"{MODEL_BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        8_373_440,
        "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
    ),
    "blaze_face_short_range.tflite": (
        f"{MODEL_BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        229_746,
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    ),
    "blaze_face_full_range.tflite": (
        f"{MODEL_BASE}/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
        1_083_786,
        "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b",
    ),
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: str, size: int, expected_hash: str) -> bool:
    return (
        os.path.isfile(path)
        and os.path.getsize(path) == size
        and sha256(path) == expected_hash
    )


def download(url: str, path: str, size: int, expected_hash: str) -> None:
    temporary = path + ".part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "hand-control-portable/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        actual_size = os.path.getsize(temporary)
        actual_hash = sha256(temporary)
        if actual_size != size or actual_hash != expected_hash:
            raise RuntimeError(
                f"校验失败：大小 {actual_size}/{size}，SHA-256 {actual_hash}/{expected_hash}"
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    failed = 0
    for name, (url, size, expected_hash) in MODELS.items():
        path = os.path.join(MODELS_DIR, name)
        if is_valid(path, size, expected_hash):
            print(f"[models] {name} 已存在且校验通过")
            continue

        print(f"[models] 下载 {name} ({size / 1024 / 1024:.1f} MB)")
        for attempt in range(1, 4):
            try:
                download(url, path, size, expected_hash)
                print(f"[models] {name} 下载并校验完成")
                break
            except Exception as error:
                print(f"[models] 第 {attempt}/3 次失败：{error}")
                if attempt == 3:
                    failed += 1
                else:
                    time.sleep(attempt * 2)

    if failed:
        print(f"[models] {failed} 个模型下载失败，请检查网络后重新运行 setup.bat")
        return 1
    print("[models] 全部模型就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())

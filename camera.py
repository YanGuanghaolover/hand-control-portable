# -*- coding: utf-8 -*-
"""
摄像头打开 —— 手势控制（main.py）和手势画图（draw.py）共用。

单独抽出来的原因：这里面全是"踩坑换来的经验"，两个程序各写一份必然会分叉。
对着 HUD 调好的后端、分辨率、超时必须完全一致，否则同一台机器上会出现
"命令行能跑、画图打不开摄像头"这种最难查的问题。
"""

from __future__ import annotations

import time

import cv2


def open_camera(index: int, w: int = 1280, h: int = 720):
    """
    按「后端 × 分辨率」逐个尝试，**实测帧率达标才用**。

    实测这台机器：MSMF@720p 正常 30FPS，但偶尔会打不开；一旦降级到
    DSHOW@720p 就只有 1–10 FPS，等于程序废掉。所以不能只看"能打开"，
    还要量一下真实帧率；DSHOW 则退到 640x480 才跑得动。

    返回 cv2.VideoCapture，或 None（全部组合都失败）。
    """
    cands = [
        (cv2.CAP_MSMF, "MSMF", w, h),
        (cv2.CAP_MSMF, "MSMF", 640, 480),
        (cv2.CAP_DSHOW, "DSHOW", 640, 480),
        (cv2.CAP_DSHOW, "DSHOW", w, h),
    ]
    for backend, name, candidate_w, candidate_h in cands:
        cap = None
        keep_open = False
        try:
            cap = cv2.VideoCapture()
            # 超时必须在 open 之前设：摄像头被别的程序占用时，read() 会一直阻塞。
            # MSMF 认这两个属性，DSHOW 不认（设了也无害）。
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            except (cv2.error, OSError):
                pass

            cap.open(index, backend)
            if not cap.isOpened():
                print(f"[camera] {name} {candidate_w}x{candidate_h}: 打开失败")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, candidate_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, candidate_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.read()[0]:
                print(f"[camera] {name} {candidate_w}x{candidate_h}: 读帧失败")
                continue

            # 启动阶段任何一帧失败都不能再按“满 10 帧”计算帧率。
            for _ in range(3):
                if not cap.read()[0]:
                    raise RuntimeError("启动阶段读帧失败")
            started = time.perf_counter()
            measured = 0
            for _ in range(10):
                if not cap.read()[0]:
                    break
                measured += 1
            elapsed = time.perf_counter() - started
            fps = measured / max(1e-6, elapsed)

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if measured == 10 and fps >= 8.0:
                print(f"[camera] {name} {actual_w}x{actual_h} @ {fps:.1f} FPS")
                keep_open = True
                return cap
            print(
                f"[camera] {name} {actual_w}x{actual_h} 有效帧 {measured}/10，"
                f"{fps:.1f} FPS，换下一个组合"
            )
        except (cv2.error, OSError, RuntimeError) as exc:
            print(f"[camera] {name} {candidate_w}x{candidate_h}: {exc}")
        finally:
            if cap is not None and not keep_open:
                cap.release()
    return None

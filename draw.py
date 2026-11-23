# -*- coding: utf-8 -*-
"""
手势画图（Air Drawing）—— 捏住食指在空中画，松手抬笔，按 s 存 PNG。

跑：
    python draw.py
    python draw.py --hand right --brush 8      # 只认右手、笔画粗一点
    python draw.py --no-window --frames 30     # 自检：不弹窗，跑 30 帧退出

按键：
    q / ESC  退出        c  清空        u  撤销
    s        保存 PNG    b  换色        e  橡皮        d  骨架开关

—— 为什么这个项目里能做，而「用手势控制鼠标」不能
    鼠标要 1px 级定位，那是手指尖这个信号给不了的；画笔要的是「看起来是我画的」，
    抖动滤掉、轨迹连贯就够了。两者的精度要求差一个量级。
    另外画笔只需要**相对轨迹**，不需要绝对定位——手往旁边挪一点，画出来的线
    整体平移，但形状是对的。这正好绕开了悬空定位最弱的一环。

—— 三层，各管一件事
    OneEuro     压抖。固定强度的平滑必然在「静止要稳」和「移动要跟手」之间二选一，
                1€ 按速度自动调截止频率：慢时滤得狠，快时几乎不滤。
    PenGate     捏合 → 落笔，双阈值滞回。单阈值会让笔迹在阈值附近断成一串虚线，
                而且每断一次就多一个 stroke，撤销要按好几次。
    InkCanvas   笔迹存独立图层。画线只做增量，每帧只合成一次，不重画历史——
                直接画在 frame 上的话，画满一屏就是几万个点每帧重绘，帧率会一路掉。

—— 不用手势分类模型
    这里只需要「捏合」这一个判据，而捏合从来都是几何算的（模型的 8 个类别里
    没有 pinch，捏合和握拳在它眼里都是 Closed_Fist）。所以直接用 HandLandmarker，
    省掉分类头的那几毫秒。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from camera import open_camera
from face_guard import FaceGuard
from gestures import analyze_two_hands, fingers_up, pinch_affinity, pinch_ratio
from hand_tracker import INDEX_TIP, HandTracker, assign_sides, draw_hand

# HUD 复用独立的中文排版模块，避免画图程序导入整个主控制流程。
from hud import TextCache, put_cn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
DEFAULT_FACE_MODEL = os.path.join(BASE_DIR, "models", "blaze_face_short_range.tflite")

# 调色板（BGR）。第一色选橙黄：它在深色摄像头画面上最扎眼，
# 而白色在亮背景（白墙、窗口）上基本看不见。
PALETTE: List[Tuple[Tuple[int, int, int], str]] = [
    ((60, 220, 255), "橙黄"),
    ((90, 255, 140), "青绿"),
    ((255, 165, 70), "天蓝"),
    ((90, 90, 255), "朱红"),
    ((245, 245, 245), "白"),
]


# ------------------------------------------------------------------ 1€ 滤波
class OneEuro:
    """
    1€ 滤波器（Casiez et al., CHI 2012）。

    为什么非它不可：普通指数平滑的平滑力度是固定的，于是只有两个选择——
    压得狠，静止时笔尖稳但快速画线明显拖后腿（笔迹落后手指一两厘米，像在水里写）；
    放得松，跟手了但静止时笔尖像筛糠，想画个圆点都画不圆。

    1€ 的思路是**按速度调截止频率**：
        手慢（抖动为主） → 截止频率低 → 平滑强 → 抖不动
        手快（有意移动） → 截止频率高 → 几乎不平滑 → 笔迹贴着指尖走
    这正好是画图需要的。

    参数（像素坐标，1280×720 下标定）：
        min_cutoff  静止时的截止频率。越小越稳、但起笔和急转弯越黏。默认 1.2
        beta        速度对截止频率的加成。越大越跟手、越大越抖。默认 0.006
        d_cutoff    速度估计自身的平滑。一般不用动。

    实现上 x/y 各自独立滤（cutoff 是二维向量），但只用一个滤波器实例——
    两条轴共用时间戳和状态，少一个对象少一次 Python 调用。
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.006,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[np.ndarray] = None
        self._dx = np.zeros(2, np.float32)
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-3))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._x is None:
            self._x = x.copy()
            self._t = t
            return x

        dt = float(t - self._t)
        if dt <= 1e-4:
            # 同一毫秒内重复调用（自检模式会这样）：不更新，直接返回上次结果
            return self._x.copy()
        self._t = t

        dx = (x - self._x) / dt
        self._dx += self._alpha(np.array([self.d_cutoff, self.d_cutoff], np.float32), dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = self._x + a * (x - self._x)
        return self._x.copy()

    def reset(self) -> None:
        self._x = None
        self._dx[:] = 0.0
        self._t = 0.0


# ------------------------------------------------------------ 捏合 -> 落笔
class PenGate:
    """
    捏合 → 落笔，松开 → 抬笔。核心是**双阈值滞回**。

    pinch_ratio 是一帧一个的估计值，在阈值附近抖是常态（同一姿势实测分布
    0.08~0.40，握拳 0.39~0.78）。单阈值会让落笔状态一帧一变：
    画出来的线断成一串小短线，每断一次还多一个 stroke，撤销要按七八次。

        on   0.30   小于它才算「捏住」。比 is_pinch 的 0.45 更严，起笔更笃定
        off  0.52   大于它才算「松开」。张开手掌时比值 > 1，一定过线
        中间        保持上一状态不变 —— 这段就是抖动的缓冲带

    另外两个细节：

        confirm = 2   连续两帧满足才落笔。单帧误判（握拳时拇指蹭了一下食指）
                      不会凭空在画布上戳一个点。
        grace = 0.18  手短暂丢检不抬笔。否则捏着画的时候 MediaPipe 丢一帧，
                      就得重新捏一次，一笔能断成七八截。

    再叠一条 affinity 判据（拇指更靠食指还是中指）：捏合 ~0.13，握拳 ~0.73。
    只看距离的话，紧握拳时拇指也贴在食指上，间距一样小。
    """

    def __init__(self, on: float = 0.30, off: float = 0.52, aff_max: float = 0.55,
                 confirm: int = 2, grace: float = 0.18) -> None:
        if not 0 < on < off:
            raise ValueError("pinch thresholds must satisfy 0 < on < off")
        if confirm < 1 or grace < 0:
            raise ValueError("confirm must be positive and grace must be non-negative")
        self.on, self.off = on, off
        self.aff_max = aff_max
        self.confirm = confirm
        self.grace = grace
        self.down = False
        self._n = 0
        self._seen = -1e9

    def update(self, ratio: Optional[float], aff: Optional[float], now: float) -> bool:
        if ratio is None or aff is None:
            # 手不在画面：短暂丢失先保持，超过 grace 才算松手
            if self.down and now - self._seen > self.grace:
                self.down = False
                self._n = 0
            return self.down

        self._seen = now

        if self.down:
            if ratio > self.off:
                self.down = False
                self._n = 0
            return self.down

        if ratio < self.on and aff < self.aff_max:
            self._n += 1
            if self._n >= self.confirm:
                self.down = True
                self._n = 0
        else:
            self._n = 0
        return self.down

    def reset(self) -> None:
        self.down = False
        self._n = 0
        self._seen = -1e9


# ------------------------------------------------------------------ 画布
class InkCanvas:
    """
    笔迹图层。两张同尺寸图：color（BGR）+ mask（单通道，0~255 不透明度）。

    为什么不是一张 BGRA：
        AA 线在 4 通道图上会连颜色一起插值——笔迹边缘的 BGR 被背景的纯黑拉暗，
        合成后一圈灰边。拆成「纯色图 + 不透明度图」就没这问题：
        color 用 LINE_8 铺满纯色，mask 用 LINE_AA 做柔边，合成时只取 mask 当 alpha。

    为什么要有独立图层（而不是每帧往 frame 上重画）：
        frame 每帧都被新画面覆盖，笔迹必须重画。画满一屏时那是几万个点，
        每帧重绘会让帧率一路掉。这里是累积层：画线只做增量，每帧一次合成。

    为什么橡皮能擦干净：
        橡皮在 mask 上画 0，直接在 alpha 上抹掉，color 里的残色不参与合成。
        如果只有一张三通道图，「擦除」只能画黑线，在白底导出时会露出黑道子。
    """

    JUMP = 60      # 起笔第一段超过这么多像素就当跳变处理，见 extend()

    def __init__(self, w: int, h: int, brush: int = 5) -> None:
        if w < 1 or h < 1 or brush < 1:
            raise ValueError("canvas width, height and brush size must be positive")
        self.w, self.h = w, h
        self.color = np.zeros((h, w, 3), np.uint8)
        self.mask = np.zeros((h, w), np.uint8)
        self.strokes: List[Tuple[list, tuple, int, bool]] = []
        self.brush_color: Tuple[int, int, int] = PALETTE[0][0]
        self.brush = brush
        self.erasing = False
        self.bb: Optional[List[int]] = None
        self._cur: Optional[list] = None
        self._last: Optional[Tuple[int, int]] = None

    # ---------------- 状态
    @property
    def active(self) -> bool:
        return self._cur is not None

    @property
    def n_strokes(self) -> int:
        return len(self.strokes)

    # ---------------- 内部
    @staticmethod
    def _ipt(p) -> Tuple[int, int]:
        return int(round(float(p[0]))), int(round(float(p[1])))

    def _grow(self, x0: int, y0: int, x1: int, y1: int) -> None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.w, x1), min(self.h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        if self.bb is None:
            self.bb = [x0, y0, x1, y1]
        else:
            b = self.bb
            b[0], b[1] = min(b[0], x0), min(b[1], y0)
            b[2], b[3] = max(b[2], x1), max(b[3], y1)

    def _line(self, p0, p1, color, width: int, erase: bool) -> None:
        col = (0, 0, 0) if erase else color
        m = 0 if erase else 255
        cv2.line(self.color, p0, p1, col, width, cv2.LINE_8)
        cv2.line(self.mask, p0, p1, m, width, cv2.LINE_AA)
        r = width // 2 + 3
        self._grow(min(p0[0], p1[0]) - r, min(p0[1], p1[1]) - r,
                   max(p0[0], p1[0]) + r, max(p0[1], p1[1]) + r)

    def _dot(self, point, color, width: int, erase: bool) -> None:
        """画一个点；起笔和撤销重绘共用，避免点状笔迹丢失。"""
        x, y = point
        radius = max(1, width // 2)
        cv2.circle(self.color, point, radius,
                   (0, 0, 0) if erase else color, -1, cv2.LINE_8)
        cv2.circle(self.mask, point, radius, 0 if erase else 255, -1, cv2.LINE_AA)
        radius += 3
        self._grow(x - radius, y - radius, x + radius, y + radius)

    # ---------------- 对外
    def begin(self, p) -> None:
        """起笔：只记起点，不画。第一段线由 extend 补，
        这样落笔瞬间手抖出来的那个尖刺不会变成一条从别处飞过来的线。"""
        self.end()
        pt = self._ipt(p)
        self._cur = [pt]
        self._last = pt
        self.strokes.append((self._cur, self.brush_color, self.brush, self.erasing))

    def extend(self, p) -> None:
        if self._cur is None:
            self.begin(p)
            return
        pt = self._ipt(p)
        if self._last is not None:
            d = max(abs(pt[0] - self._last[0]), abs(pt[1] - self._last[1]))
            if d < 1:
                return      # 手指基本没动，不必生成线段（也省内存）
            if len(self._cur) == 1 and d > self.JUMP:
                # 起笔第一段的大跳变，跳过不画。
                # 「捏合」这个动作本身会把食指尖往掌心带（伸直的食指一弯，
                # 指尖能挪几十像素），而落笔判定要等两帧才确认 —— 于是落笔点
                # 和第二帧的位置差出一大截。照实画就是一条无缘无故的飞线。
                # 处理成「把起点挪到当前位置」：笔迹从稳定处开始，不丢内容。
                # 顺手 append 一个同点，让 len 变成 2 —— 否则下一段还会被判成
                # 「第一段」而继续跳过，整笔就永远画不出来。
                self._cur[0] = pt
                self._cur.append(pt)
                self._last = pt
                return
        self._line(self._last, pt, self.brush_color, self.brush, self.erasing)
        self._cur.append(pt)
        self._last = pt

    def end(self) -> None:
        # 只捏了一下就松开（整笔只有起点、没来得及产生线段）时补一个点。
        # 不补的话"点一下"什么都不会出现 —— 而点画是画图里最基本的动作之一。
        if self._cur is not None and len(set(self._cur)) == 1:
            pts, col, wd, er = self.strokes[-1]
            self._dot(pts[0], col, wd, er)
        self._cur = None
        self._last = None

    def undo(self) -> bool:
        self.end()
        if not self.strokes:
            return False
        self.strokes.pop()
        self._redraw()
        return True

    def clear(self) -> None:
        self.end()
        self.strokes.clear()
        self.color[:] = 0
        self.mask[:] = 0
        self.bb = None

    def _redraw(self) -> None:
        self.color[:] = 0
        self.mask[:] = 0
        self.bb = None
        for pts, col, wd, er in self.strokes:
            if len(pts) == 1:
                self._dot(pts[0], col, wd, er)
                continue
            for a, b in zip(pts, pts[1:]):
                self._line(a, b, col, wd, er)

    def composite(self, frame: np.ndarray, alpha: float = 0.9) -> None:
        """把笔迹半透明叠到 frame 上，只处理有墨的 bbox。"""
        if self.bb is None:
            return
        x0, y0, x1, y1 = self.bb
        if x1 <= x0 or y1 <= y0:
            return
        m = self.mask[y0:y1, x0:x1]
        if not m.any():
            return
        a = m.astype(np.float32)[:, :, None] * (alpha / 255.0)
        roi = frame[y0:y1, x0:x1]
        f = roi.astype(np.float32)
        roi[:] = (self.color[y0:y1, x0:x1].astype(np.float32) * a + f * (1.0 - a)).astype(np.uint8)

    def flatten(self, bg: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """导出用：笔迹压在纯色底上，返回 BGR 图。"""
        a = (self.mask.astype(np.float32) / 255.0)[:, :, None]
        b = np.array(bg, np.float32).reshape(1, 1, 3)
        return (self.color.astype(np.float32) * a + b * (1.0 - a)).astype(np.uint8)


# ------------------------------------------------------------ 双手张开清空
class BothOpenHold:
    """
    双手同时张开手掌、稳住 hold 秒 → 清空画布。

    为什么清空要用双手，不挂在单手上：
        画图时右手一直被「捏合 = 落笔」占着，再给它别的手势必然打架。
        双手张开是个不会在画图过程中出现的姿态（画的时候至少有一只手是捏着的），
        所以拿它当"开保险箱"动作是安全的。

    为什么按住 1 秒而不是挥一下：
        挥过去太容易误触，而清空是不可撤销的操作（虽然能叠加 undo，但画了一半
        的东西没了就是没了）。
    """

    def __init__(self, hold: float = 1.0) -> None:
        self.hold = hold
        self._since: Optional[float] = None
        self.progress = 0.0

    def update(self, both_open: bool, now: float) -> bool:
        if not both_open:
            self._since = None
            self.progress = 0.0
            return False
        if self._since is None:
            self._since = now
        self.progress = min(1.0, (now - self._since) / max(self.hold, 1e-6))
        if self.progress >= 1.0:
            self._since = None
            self.progress = 0.0
            return True
        return False


# ------------------------------------------------------------------ 工具
def save_png(img: np.ndarray, path: str) -> bool:
    """存图。走 imencode + tofile 而不是 imwrite：后者在中文路径下静默失败。"""
    d = os.path.dirname(os.path.abspath(path))
    try:
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception as e:
        print(f"[error] 建目录失败 {d}: {e}")
        return False
    ok, buf = False, None
    try:
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    except (cv2.error, OSError, ValueError) as e:
        print(f"[error] 编码图片失败 {path}: {e}")
        return False
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception as e:
        print(f"[error] 写文件失败 {path}: {e}")
        return False


def pick_hand(st, prefer: str):
    """选画笔手。any 时优先右手（项目的主力手），没有就用左手。"""
    if prefer == "left":
        return st.left
    if prefer == "right":
        return st.right
    return st.right or st.left


def parse_args() -> argparse.Namespace:
    from cli import non_negative_float, non_negative_int, positive_float, positive_int

    p = argparse.ArgumentParser(description="手势画图：捏住食指空中作画（MediaPipe + 1€ 滤波）")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cam", type=non_negative_int, default=0)
    p.add_argument("--width", type=positive_int, default=1280)
    p.add_argument("--height", type=positive_int, default=720)
    p.add_argument("--hand", choices=["any", "left", "right"], default="any",
                   help="用哪只手画（默认 any：有右手用右手，否则左手）")
    p.add_argument("--on", type=positive_float, default=0.30,
                   help="落笔阈值：拇指尖-食指尖距离/掌长 小于它才算捏住（默认 0.30）")
    p.add_argument("--off", type=positive_float, default=0.52,
                   help="抬笔阈值：大于它才算松开（默认 0.52）。必须比 --on 大，中间是缓冲带")
    p.add_argument("--min-cutoff", type=positive_float, default=1.2,
                   help="1€ 滤波静止截止频率，越小笔尖越稳、起笔越黏（默认 1.2）")
    p.add_argument("--beta", type=non_negative_float, default=0.006,
                   help="1€ 滤波速度加成，越大越跟手、越大越抖（默认 0.006）")
    p.add_argument("--no-smooth", action="store_true", help="关掉滤波（笔迹会抖，用来对比）")
    p.add_argument("--brush", type=positive_int, default=5, help="笔宽像素（默认 5）")
    p.add_argument("--out", default=os.path.join(BASE_DIR, "drawings"),
                   help="保存目录（默认 ./drawings）")
    p.add_argument("--no-face-guard", action="store_true",
                   help="关掉人脸否决（默认开启：脸被当成手会让笔尖突然跳到脸上画一笔）")
    p.add_argument("--face-model", default=DEFAULT_FACE_MODEL)
    p.add_argument("--no-window", action="store_true", help="不开窗口（自检用，配 --frames）")
    p.add_argument("--frames", type=positive_int, default=40, help="自检跑多少帧")
    p.add_argument("--snapshot", default=None, help="自检：跑完存一张图")
    p.add_argument("--debug", action="store_true", help="HUD 显示捏合比等原始判据")
    return p.parse_args()


# ------------------------------------------------------------------ 主流程
def main() -> int:
    args = parse_args()

    if args.off <= args.on:
        print(f"[error] --off({args.off}) 必须大于 --on({args.on})，否则滞回不成立")
        return 1
    if not os.path.exists(args.model):
        print(f"[error] 手部模型不存在: {args.model}")
        return 1

    print(f"[model] 加载 {args.model}")
    t0 = time.perf_counter()
    # 不传 gesture_model：画图只用捏合（几何判据），不需要手势分类头
    tracker = HandTracker(args.model, num_hands=2, hand_mode="auto")
    print(f"[model] 就绪 {time.perf_counter() - t0:.2f}s")

    face_guard = None
    if not args.no_face_guard:
        face_guard = FaceGuard(args.face_model)
        if face_guard.enabled:
            print("[face] 人脸否决已启用")

    cap = open_camera(args.cam, args.width, args.height)
    if cap is None:
        print(f"[error] 打不开摄像头 {args.cam}")
        tracker.close()
        if face_guard is not None:
            face_guard.close()
        return 1

    tc = TextCache(18)
    if not tc.ok:
        print("[warn] 没找到中文字体，HUD 会退化成英文")

    canvas: Optional[InkCanvas] = None
    filt = OneEuro(args.min_cutoff, args.beta)
    pen = PenGate(args.on, args.off)
    clear_hold = BothOpenHold(1.0)

    show_skeleton = True
    ci = 0                        # 当前颜色索引
    last_msg, last_msg_at = "", 0.0
    fps_hist: List[float] = []
    frame_idx = 0
    read_failures = 0

    win = "Hand Drawing"
    ts = 0
    start = time.perf_counter()
    loop_end = start

    print("[draw] 捏合落笔 / 松开抬笔    双手张开 1 秒清空")
    print("[draw] 按键：q 退出  c 清空  u 撤销  s 保存  b 换色  e 橡皮  d 骨架")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                print(f"[warn] 读帧失败（连续 {read_failures}/3 次）")
                if read_failures >= 3:
                    break
                continue
            read_failures = 0

            frame = cv2.flip(frame, 1)      # 镜像：不翻的话举右手屏幕显示在左边
            h, w = frame.shape[:2]
            now = time.perf_counter()
            ts = max(ts + 1, int((now - start) * 1000))

            hands = tracker.detect(frame, ts)
            if face_guard is not None:
                hands, dropped = face_guard.filter(frame, hands)
                if dropped:
                    # 少了一只手，左右关系变了，用同一套规则重排
                    assign_sides(hands, "auto", False)

            st = analyze_two_hands(hands)

            if canvas is None:
                canvas = InkCanvas(w, h, args.brush)

            # ---- 笔尖：先取原始像素点，再滤波
            pen_hand = pick_hand(st, args.hand)
            p = None
            ratio = aff = None
            if pen_hand is not None:
                lm = pen_hand.landmarks[INDEX_TIP]
                p = filt(np.array([lm[0] * w, lm[1] * h], np.float32), now) \
                    if not args.no_smooth else np.array([lm[0] * w, lm[1] * h], np.float32)
                ratio = pinch_ratio(pen_hand)
                aff = pinch_affinity(pen_hand)

            down = pen.update(ratio, aff, now)

            # ---- 画
            if p is None:
                # 手不见了就断笔。不能等 grace 过期再断：手移出画面又移回来时
                # 位置已经变了，连上去就是一条横跨画面的直线。
                canvas.end()
            elif down:
                if not canvas.active:
                    canvas.begin(p)
                else:
                    canvas.extend(p)
            else:
                canvas.end()

            # ---- 双手张开 1 秒 -> 清空
            both_open = (st.count == 2 and all(fingers_up(st.left))
                         and all(fingers_up(st.right)))
            if clear_hold.update(both_open, now):
                canvas.clear()
                last_msg, last_msg_at = "已清空", now

            # ---------------- 绘制 ----------------
            canvas.composite(frame, alpha=0.9)

            vis = frame
            if show_skeleton:
                for hd in hands:
                    draw_hand(vis, hd, (0, 220, 120))

            # 笔尖光标：落笔绿、抬笔白，给手上一个明确的反馈
            if p is not None:
                cx, cy = int(p[0]), int(p[1])
                col = (0, 255, 120) if down else (255, 255, 255)
                cv2.circle(vis, (cx, cy), 11, col, 2, cv2.LINE_AA)
                cv2.circle(vis, (cx, cy), 2, col, -1, cv2.LINE_AA)

            # 顶部状态条
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (w, 66), (26, 28, 30), -1)
            cv2.addWeighted(overlay, 0.62, vis, 0.38, 0, vis)

            fps_hist.append(now)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            fps = ((len(fps_hist) - 1) / max(1e-6, fps_hist[-1] - fps_hist[0])
                   if len(fps_hist) > 2 else 0.0)

            put_cn(vis, "画图中" if down else "抬笔", 10, 4, tc,
                   (0, 255, 120) if down else (200, 200, 200))
            cv2.putText(vis, f"FPS {fps:4.1f}  ink {canvas.n_strokes}", (96, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (232, 232, 232), 1, cv2.LINE_AA)

            # 画笔状态：色块 + 名字 + 笔宽
            bx = 300
            cv2.rectangle(vis, (bx, 10), (bx + 22, 32), canvas.brush_color, -1)
            cv2.rectangle(vis, (bx, 10), (bx + 22, 32), (240, 240, 240), 1)
            label = "橡皮" if canvas.erasing else PALETTE[ci][1]
            put_cn(vis, label, bx + 30, 4, tc, (235, 235, 235))

            put_cn(vis, "捏合画  张开停  双手张开1秒清空", 10, 36, tc, (200, 200, 200))

            if args.debug:
                cv2.putText(vis,
                            f"p={ratio:.2f} a={aff:.2f} on={args.on:.2f} off={args.off:.2f}"
                            if ratio is not None else "no hand",
                            (500, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 230, 255), 1, cv2.LINE_AA)

            if clear_hold.progress > 0:
                bw = 140
                bx2 = w // 2 - bw // 2
                cv2.rectangle(vis, (bx2, 78), (bx2 + bw, 86), (60, 60, 64), -1)
                cv2.rectangle(vis, (bx2, 78), (bx2 + int(bw * clear_hold.progress), 86),
                              (0, 170, 255), -1)
                put_cn(vis, "松开就清空", bx2 + 30, 90, tc, (0, 170, 255))

            if last_msg and now - last_msg_at < 1.2:
                put_cn(vis, "→ " + last_msg, w - 150, 10, tc, (0, 220, 255))

            put_cn(vis, "q 退出   c 清空   u 撤销   s 保存   b 换色   e 橡皮   d 骨架",
                   10, h - 34, tc, (175, 175, 175))

            if (args.snapshot or args.no_window) and frame_idx >= args.frames:
                if args.snapshot:
                    out = args.snapshot
                    img = canvas.flatten() if canvas.n_strokes else vis
                    print(f"[snapshot] {out}" if save_png(img, out) else "[snapshot] 失败")
                break

            frame_idx += 1

            if args.no_window:
                continue

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                canvas.clear()
                last_msg, last_msg_at = "已清空", now
            elif key == ord("u"):
                last_msg, last_msg_at = ("撤销一步" if canvas.undo() else "没东西可撤销"), now
            elif key == ord("b"):
                ci = (ci + 1) % len(PALETTE)
                canvas.brush_color = PALETTE[ci][0]
                canvas.erasing = False
                last_msg, last_msg_at = "颜色 " + PALETTE[ci][1], now
            elif key == ord("e"):
                canvas.erasing = not canvas.erasing
                last_msg, last_msg_at = ("橡皮开" if canvas.erasing else "橡皮关"), now
            elif key == ord("d"):
                show_skeleton = not show_skeleton
            elif key == ord("s"):
                name = time.strftime("draw_%Y%m%d_%H%M%S.png")
                path = os.path.join(args.out, name)
                ok_save = save_png(canvas.flatten(), path)
                print(f"[save] {path}" if ok_save else "[save] 失败")
                last_msg, last_msg_at = ("已保存 " + name if ok_save else "保存失败"), now

    finally:
        loop_end = time.perf_counter()
        cap.release()
        tracker.close()
        if face_guard is not None:
            face_guard.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    dur = loop_end - start
    print(f"[done] {frame_idx} 帧 / {dur:.1f}s = {frame_idx / max(dur, 1e-6):.1f} FPS")
    if canvas is not None and canvas.n_strokes:
        print(f"[ink] {canvas.n_strokes} 笔，导出白底图请按 s")
    return 0


if __name__ == "__main__":
    _rc = main()
    # 和 main.py 同样的问题：mediapipe 的原生收尾偶发挂住，日志打完进程却不退。
    # 该释放的都在 main 里做完了，这里直接硬退出。
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(_rc)

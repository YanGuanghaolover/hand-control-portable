# -*- coding: utf-8 -*-
"""
画图模块的单元测试（不需要摄像头、不需要模型）。

跑法：
    python tests/test_draw.py

只测三个纯逻辑组件：1€ 滤波、捏合落笔门、笔迹画布。
它们决定了画出来的东西能不能看，出问题也最难从画面上反推原因
（"抖"到底是滤波没工作还是落笔门在闪，只能靠测试分开验证）。
"""
import os
import sys

import numpy as np

# CI logs on Windows are redirected and may default to cp1252; force UTF-8
# so Chinese labels can be printed without crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import draw as D                     # noqa: E402

DT = 1 / 30
PASS = []


def check(name, cond, extra=""):
    PASS.append(bool(cond))
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        raise AssertionError(name)


# ---------------------------------------------------------------- 1€ 滤波
print("== OneEuro：静止压抖 / 快速跟手 ==")
f = D.OneEuro(1.2, 0.006)
t, ys = 0.0, []
for i in range(120):
    t += DT
    raw = 100.0 + (1.5 if i % 2 else -1.5)     # ±1.5px 的逐帧抖动
    ys.append(float(f(np.array([raw, 0], np.float32), t)[0]))
sd = float(np.std(ys[40:]))
check("静止抖动被压掉 80% 以上", sd < 0.30, f"(std 1.50 -> {sd:.3f})")

f = D.OneEuro(1.2, 0.006)
t, err = 0.0, []
for i in range(90):
    t += DT
    truth = 100.0 + 12.0 * i                   # 360 px/s 匀速
    out = float(f(np.array([truth, 0], np.float32), t)[0])
    if i > 10:
        err.append(abs(out - truth))
lag = float(np.mean(err))
check("匀速 360px/s 的滞后小于 20px", lag < 20.0, f"({lag:.1f} px)")

f = D.OneEuro(1.2, 0.006)
out0 = f(np.array([10.0, 20.0], np.float32), 0.0)
check("第一帧原样输出", np.allclose(out0, [10.0, 20.0]))

# ------------------------------------------------------------ 捏合落笔门
print("== PenGate：滞回 / 确认帧 / 丢检宽容 ==")
g = D.PenGate(0.30, 0.52)
t, states = 0.0, []
for r in [1.0] * 5 + [0.15] * 20 + [0.9] * 5:
    t += DT
    states.append(g.update(r, 0.13 if r < 0.5 else 0.73, t))
first_down = states.index(True)
check("捏合后 2 帧内落笔", first_down == 6, f"(第 {first_down} 帧)")
check("张开手掌立刻抬笔", states.index(False, first_down) == 25)

# 阈值附近交替抖动：不能落笔（否则一笔断成十几个 stroke）
g = D.PenGate(0.30, 0.52)
t, n_down, prev = 0.0, 0, False
for i in range(60):
    t += DT
    now = g.update(0.31 + (0.04 if i % 2 else -0.04), 0.2, t)
    n_down += (now and not prev)
    prev = now
check("阈值抖动不落笔", n_down == 0, f"({n_down} 次)")

# 一旦落笔，回到缓冲带内（0.30~0.52）不该抬笔
g = D.PenGate(0.30, 0.52)
t = 0.0
for _ in range(10):
    t += DT
    g.update(0.15, 0.13, t)
t += DT
still = g.update(0.42, 0.4, t)          # 缓冲带中间
check("缓冲带内保持落笔", still is True)

# 手短暂丢检
g = D.PenGate(0.30, 0.52)
t = 0.0
for _ in range(10):
    t += DT
    g.update(0.15, 0.13, t)
for _ in range(3):
    t += DT
    keep = g.update(None, None, t)
check("丢检 0.1s 仍算落笔", keep is True)
for _ in range(20):
    t += DT
    keep = g.update(None, None, t)
check("丢检 0.67s 算松手", keep is False)

# 握拳（拇指横搭、affinity 高）不能落笔
g = D.PenGate(0.30, 0.52)
t, hit = 0.0, False
for _ in range(20):
    t += DT
    hit = hit or g.update(0.28, 0.73, t)    # 距离够近，但拇指贴的是中指
check("握拳不落笔", hit is False)

# ---------------------------------------------------------------- 画布
print("== InkCanvas：增量绘制 / 撤销 / 橡皮 / 导出 ==")


def stroke(c, y, x0=50, x1=150, step=5):
    """画一条水平轨迹（真实一笔有几十段，不能只给首尾两点）"""
    c.begin((float(x0), float(y)))
    for x in range(x0 + step, x1 + 1, step):
        c.extend((float(x), float(y)))
    c.end()


c = D.InkCanvas(400, 300, 5)
stroke(c, 50)
n1 = int((c.mask > 0).sum())
check("一笔画出墨", n1 > 0, f"({n1} px)")

stroke(c, 120)
n2 = int((c.mask > 0).sum())
check("两笔的墨量约为一笔的两倍", abs(n2 - 2 * n1) < n1 * 0.15, f"({n1} -> {n2})")

c.undo()
n3 = int((c.mask > 0).sum())
check("撤销后回到一笔", abs(n3 - n1) <= 2, f"({n3} vs {n1})")
check("撤销后笔画计数正确", c.n_strokes == 1)

c.undo()
check("空画布再撤销返回 False", c.undo() is False)

# 点一下（整笔只有一个点）要留下一个点，不能什么都没有
c = D.InkCanvas(200, 200, 6)
c.begin((100.0, 100.0))
c.end()
check("点一下留下一个点", int((c.mask > 0).sum()) > 0, f"({int((c.mask > 0).sum())} px)")

# 起笔跳变：第一条线段超过 JUMP 时不画，起点挪到新位置
c = D.InkCanvas(400, 300, 5)
c.begin((50.0, 200.0))
c.extend((50.0, 50.0))               # 150px 的大跳变
check("起笔跳变不落墨", int((c.mask > 0).sum()) == 0)
check("起笔跳变后起点被挪过去", c.strokes[0][0][0] == (50, 50))
c.extend((120.0, 50.0))
check("跳变之后正常画", int((c.mask > 0).sum()) > 0)

# 橡皮
c = D.InkCanvas(200, 200, 5)
stroke(c, 100, 20, 180)
before = int((c.mask > 0).sum())
c.erasing = True
c.brush = 20
stroke(c, 100, 4, 196)
check("橡皮把墨擦干净", int((c.mask > 0).sum()) == 0, f"({before} -> 0)")

# 导出白底
c = D.InkCanvas(120, 80, 6)
c.brush_color = (60, 220, 255)
stroke(c, 40, 20, 100)
flat = c.flatten()
check("导出图尺寸正确", flat.shape == (80, 120, 3))
check("空白处是白底", tuple(flat[2, 2]) == (255, 255, 255))
check("线心是画笔色", tuple(flat[40, 60]) == (60, 220, 255))

# 合成：alpha=0.9 时，线心应接近画笔色但混进一点背景
frame = np.full((80, 120, 3), 100, np.uint8)
c.composite(frame, alpha=0.9)
px = frame[40, 60]
check("合成后线心介于画笔色和背景之间",
      px[2] > 200 and px[0] < 100, f"(BGR={tuple(int(v) for v in px)})")
check("画布外的像素不受影响", tuple(frame[2, 2]) == (100, 100, 100))

# ---------------------------------------------------------------- 清空动作
print("== BothOpenHold：双手张开清空 ==")
h = D.BothOpenHold(1.0)
t, fired = 0.0, False
for _ in range(10):
    t += DT
    fired = fired or h.update(True, t)
check("不足 1 秒不触发", fired is False)
for _ in range(25):
    t += DT
    fired = fired or h.update(True, t)
check("持续 1 秒触发", fired is True)
check("触发后进度归零", h.progress == 0.0)

h = D.BothOpenHold(1.0)
t = 0.0
for _ in range(20):
    t += DT
    h.update(True, t)
t += DT
h.update(False, t)                    # 中途松开一只手
t += DT
mid = h.progress
check("中途松手则进度清零", mid == 0.0)

print(f"\n{len(PASS)} 项全部通过")

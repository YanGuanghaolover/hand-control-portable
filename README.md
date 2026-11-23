# hand-control-portable

用普通摄像头 + MediaPipe 双手手势控制 Windows：播放/暂停、音量、切窗口、翻页、缩放、滚动，以及在空中用手势画图。纯 CPU 实时，无需训练模型，也无需显卡。

> **第一次使用请先跑 `run.bat --test`**：只识别手势，不向系统发送任何按键。

## 快速开始

要求 Windows 10/11 64 位、Python 3.11–3.13、任意摄像头。

```bash
git clone https://github.com/YanGuanghaolover/hand-control-portable.git
cd hand-control-portable
```

然后双击 `setup.bat`。它会：

1. 自动挑选可用的 Python 3.11–3.13；
2. 创建 `.venv`，已损坏的旧环境会重建；
3. 只走官方 PyPI 安装 `mediapipe / opencv / numpy / pillow`；
4. 调用 `download_models.py` 下载模型，逐个校验大小与 SHA-256。

模型二进制不入库，首次安装约需 200 MB 依赖 + 17 MB 模型。装好后双击 `run.bat`，或：

```bash
run.bat --test        # 只识别，不发送按键（建议先跑）
run.bat               # 正常控制
run_draw.bat          # 手势画图
```

## 两个入口

| 入口 | 用途 |
|---|---|
| `run.bat` / `python main.py` | HUD 主程序：摄像头画面、骨架、手势图例、实时状态 |
| `run_draw.bat` / `python draw.py` | 手势画图：拇指与食指捏合落笔，松手抬笔，`s` 保存 PNG |

## 手势

单手指令需要悬停约 0.5 秒，手在移动时计时冻结，触发后有冷却，避免挥手误触。

| 手势 | 右手 | 左手 |
|---|---|---|
| 张开手掌 | 播放/暂停 | 播放/暂停 |
| 握拳 | 静音 | 静音 |
| 竖拇指 / 倒拇指 | 音量 + / − | 音量 + / − |
| 剪刀手 | 切换窗口 | 切换窗口 |
| 食指 | 下一页 | 上一页 |

双手动作：

- 双手捏合后拉开/收拢：网页缩放（桌面和资源管理器自动跳过，保护图标大小）；
- 双手捏合连捏两次：缩放恢复 100%；
- 双手张开同时上下移动：页面滚动；
- 两手同时「握拳 → 张开」连做两遍：开启/关闭全部控制（紧急停止）。

`run.bat --debug` 显示五指状态与捏合比，`run.bat --trace` 在退出时统计手势为什么没触发。

## 稳定性重构

这个版本在原功能不变的前提下，集中修复了资源泄漏、异常路径和参数边界：

- **摄像头候选**：每个「后端 × 分辨率」组合独立 `try/finally`，失败候选一定 `release()`；启动阶段要求连续读到有效帧，失败帧不计入成功。
- **模型释放**：`HandTracker.close()` 与 `FaceGuard.close()` 幂等，重复调用只释放一次；MediaPipe 原生收尾放到后台线程，最多等 0.5 秒。
- **组合键异常**：`actions.combo()` 用 `try/finally` 保证 Alt/Ctrl 等修饰键一定会抬起，不会卡在按下状态。
- **画布撤销**：起笔点画与撤销重绘共用同一条绘制路径，修复「点状笔迹撤销后消失」。
- **保存失败**：`save_png()` 捕获编码异常并返回失败，非法扩展名不再让程序崩溃。
- **时间戳**：主程序、画图、校准都不再用固定 `+33ms`，改为真实经过时间；短暂读帧失败容忍 3 次。
- **校准清理**：采集循环用 `try/finally`，异常退出也释放摄像头和 tracker；输出路径改为绝对路径。
- **参数校验**：统一在 `cli.py` 拒绝 0/负数分辨率、非法概率和非正超时；状态机组件在构造时检查非法参数。
- **等比坐标**：掌长与双手距离统一使用按画面宽高比校正的坐标，16:9 下不再压扁横向距离。
- **可测试性**：新增 `hud.py` 与 `cli.py`，把 HUD 渲染和参数校验从主程序解耦。

## 测试

不需要摄像头和模型：

```bash
python -m unittest discover -s tests -v
```

覆盖手势状态机、1€ 滤波、捏合落笔、笔迹撤销/橡皮/导出、双手距离，以及摄像头候选释放、重复 close、组合键异常释放、保存异常和参数校验。GitHub Actions 在 Windows + Python 3.13 上执行同一套测试。

## 目录

```text
hand-control-portable/
├── setup.bat / run.bat / run_draw.bat   安装与启动
├── main.py                              HUD 主程序
├── draw.py                              手势画图
├── hand_tracker.py                      MediaPipe 双手 21 点封装
├── face_guard.py                        人脸否决，剔除「长在脸上的幻影手」
├── gestures.py                          手势规则、防抖、缩放步进
├── hud.py                               共用中文 HUD 渲染
├── cli.py                               共用参数校验
├── actions.py                           Windows 按键/滚轮注入
├── camera.py                            摄像头打开与候选回退
├── calibrate.py                         手势自测/校准
├── download_models.py                   下载模型并校验 SHA-256
├── models/                              模型目录（二进制首次自动下载）
├── docs/START_HERE.txt                  给非开发者的上手说明
└── tests/                               单元测试
```

## 模型

`download_models.py` 从 MediaPipe 官方 `storage.googleapis.com/mediapipe-models` 下载四个文件，先写 `.part`，校验精确大小和 SHA-256，通过后才原子替换。已有且校验通过的文件会跳过；单个文件最多重试 3 次。

- `hand_landmarker.task`
- `gesture_recognizer.task`
- `blaze_face_short_range.tflite`
- `blaze_face_full_range.tflite`

默认只使用 short-range 人脸模型；full-range 文件预置给需要更远检测的场景。

## 已知限制

- 不要用它控制鼠标指针：指尖悬空定位精度不足；画图只用相对轨迹，所以不受此限。
- 手部遮挡、极端角度会丢检，依靠 0.35 秒宽容窗口和防抖兜底。
- 双手交叉时位置判定失效，需要 `--hands model`。
- 手贴着脸或挡着脸做手势会被人脸否决忽略。
- 画图范围受摄像头视野限制，也写不了小于约 1 cm 的小字。

更详细的参数、调参方法和踩坑记录见 [`docs/START_HERE.txt`](docs/START_HERE.txt) 与源码注释。

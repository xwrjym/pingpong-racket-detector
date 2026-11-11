# 球拍检测模型（独立工程）

仓库：https://github.com/xwrjym/pingpong-racket-detector

对应方案：`球拍检测模型独立方案.md`。本工程实现方案里的 **路线 A：独立球拍模型 + 独立适配器 + Proposal 区间触发**，
不改球模型、不改计分状态机；球拍证据只交给融合层做 `HIT_CANDIDATE`，最终判分仍由现有状态机负责。

## 0. 交付约定（用户要求，务必遵守）

- **所有交付物默认放到桌面**：`C:\Users\rokmy\Desktop`，不要只留在工程目录里让用户自己找。
- 每次交付至少包括：报告（Markdown）、模型权重、演示视频（如有）、报告引用的图片。
- 报告里的图片用**相对路径**引用，图片文件必须和 `.md` 一起放在桌面，否则发出去看不到图。
- DSH 沙箱只允许写 `D:\work`，往桌面复制需要一次性提权（`danger-full-access`），这是预期行为。

## 1. 实现了方案的哪些部分

| 方案章节 | 落地文件 | 说明 |
|---|---|---|
| §5 数据集与标注 | `racket/dataset.py` | 帧级标注 + 困难标签 + 击球区间；**按场次切分**，同回合不跨集合 |
| §6 统一输出接口 | `racket/schema.py` | `racket.v1` 契约，字段与方案逐字段对应，出口强校验 |
| §3 球拍证据 | `racket/tracker.py` | 侧别、位置、速度、方向、track_id、断轨统计 |
| §2 Proposal 触发 | `racket/adapter.py` (`ProposalGate`) | 只在提议区间内推理，区间外不调用模型 |
| §7 融合 | `racket/fusion.py` | `HIT_score` 基线、证据串、`quality_score` |
| §12.3 非模型证据 | `racket/fusion.py` (`is_suspect_hit`) | `suspect_hit` 影子标记，默认不否决 HIT |
| §12.4 降级 | `racket/fusion.py` (`score_hit`) | 球拍不可用时退化并显式标记 |
| §10 验收指标 | `racket/eval.py` | mAP/P/R、击球时刻召回、侧别准确率、区间覆盖、断轨 |
| §12.2 止损检查点 | `racket/tools/selfcheck.py`、`racket/eval.py` | 小样本先验证，指标不达标就别加量 |
| §8 工作包 | `racket/train.py`、`racket/detectors/` | 训练、ONNX 导出、板端推理后端 |

## 2. 目录

```text
racket/
  schema.py       racket.v1 契约（观测 / 击球候选 / JSONL 日志）
  geometry.py     框几何、侧别判定（压网线判 UNKNOWN，不猜）
  tracker.py      轻量跟踪器：IoU + 匀速预测，输出速度/方向/断轨
  dataset.py      标注格式、按场次切分、YOLO 导出
  prelabel.py     半自动预标注（帧差 + 圆检测，无需权重）
  adapter.py      检测后端 -> 观测；Proposal 门控；坐标时间校验
  fusion.py       HIT_score 基线、suspect_hit、球拍缺失降级
  eval.py         检测/击球/区间/跟踪指标 + Markdown 报告
  train.py        训练、测试集评估、ONNX 导出
  detectors/      yolo（ultralytics）、onnx（板端）、fake（无依赖演练）
  tools/
    extract_frames.py  抽帧（opencv 或 ffmpeg）
    synthetic.py       合成画面（仅供自检，不能当精度）
    selfcheck.py       全链路自检
    annotate.py        浏览器标注复核服务
    annotator.html     标注页面
tests/            49 项单测
```

## 3. 环境

沙箱内装不了依赖，请在**普通 PowerShell**里执行一次：

```powershell
py -3.11 -m venv D:\racket-env
D:\racket-env\Scripts\python.exe -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
D:\racket-env\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
D:\racket-env\Scripts\python.exe -m pip install ultralytics imageio-ffmpeg opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
D:\racket-env\Scripts\python.exe -c "import torch,cv2,ultralytics;print(torch.cuda.is_available())"
```

没有 GPU 就把第二行换成纯 CPU：`pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple`，训练慢 5–8 倍，功能一样。

## 4. 从视频到模型的完整流程

```powershell
cd D:\work\racket
$PY = "D:\racket-env\Scripts\python.exe"

# 1) 抽帧（stride 5 ≈ 6 fps；先抽一小段做止损检查点）
& $PY -m racket.tools.extract_frames --video D:\racket-data\s01.mp4 --out D:\racket-data\frames\s01 --stride 5

# 2) 生成 manifest（filename 的数字即 frame_id）
& $PY -c "from racket.prelabel import build_manifest; print(build_manifest(r'D:\racket-data\frames\s01', r'D:\racket-data\manifest_s01.jsonl', session_id='s01', fps=6, image_width=1920, image_height=1080))"

# 3) 半自动预标注（无权重也能跑）
& $PY -c "from racket.prelabel import prelabel_manifest; print(prelabel_manifest(r'D:\racket-data\manifest_s01.jsonl', r'D:\racket-data\labels_s01.jsonl', r'D:\racket-data\prelabel_meta.jsonl'))"

# 4) 人工复核（浏览器打开 http://127.0.0.1:8710）
& $PY -m racket.tools.annotate --labels D:\racket-data\labels_s01.jsonl --port 8710

# 5) 合并多场标注后训练（至少 3 个场次才能切出 train/val/test）
& $PY -m racket.train --labels D:\racket-data\labels_all.jsonl --dest D:\racket-data\yolo --epochs 120 --imgsz 640 --batch 8 --export-onnx
```

训练完输出 `runs/racket_v0.1/weights/best.pt`，同时打印测试集 mAP。

## 5. 推理与接入

**推荐配置：`weights/racket_v11_两场地.pt` + imgsz 1280 + conf 0.5**（覆盖两家球馆，含 v8 权重仅用于对照）。

```python
from racket.adapter import AdapterConfig, ProposalGate, RacketAdapter
from racket.detectors import YoloRacketDetector
from racket.schema import ObservationLog

detector = YoloRacketDetector(r"weights/racket_v11_两场地.pt", imgsz=1280, conf_thr=0.5)
adapter = RacketAdapter(
    detector,
    AdapterConfig(session_id="s01", source="racket_model_a", model_version="racket_v1.0", conf_thr=0.5),
    gate=ProposalGate([(15.0, 15.6)]),          # TrackNet 提议区间，可留空=全程
    log=ObservationLog("runtime/racket_obs.jsonl", truncate=True),
)
for frame, frame_id, ts, size in frames:        # 你自己的取帧循环
    observations = adapter.process_frame(frame, frame_id, ts, size)
adapter.close()
```

输出的每条观测都是 `racket.v1`，可直接拼进 `tt_referee` 的适配器链路；
`racket.eval.from_observations()` 能把日志转成评估输入。

## 6. 当前验证状态

```powershell
& $PY -m unittest discover -s tests -t .      # 51 项
& $PY -m racket.tools.selfcheck --out reports\selfcheck.md
```

- 单测 **51 项通过**（契约、几何、跟踪、数据集防泄漏、指标方向性、适配器边界、同侧去重）；
- 真实数据：6 个场次、629 帧、约 960 个球拍框（s01/s02/s04/s06 训练，s03 整场留出）；
- **同场地、未见过的选手**：mAP@0.5 0.692（COCO）/ 0.591（部署链路），召回 0.60；
- **换到没见过的场地**：mAP@0.5 0.318 / 0.267，精确率从 0.79 掉到 0.45。

## 7. 部署与场地标定（重要）

实测结论：**这个模型对"选手"泛化，对"场地"过拟合**。训练集只有一家球馆时，
换到另一家球馆精确率会从 0.79 掉到 0.45（背景里的拍形 LOGO 被大量误报）。

因此现场部署按下面的流程走：

1. **目标场地先标 70–100 帧**（半自动预填 + 人工复核，约 15 分钟）；
2. 把这几帧并入训练集重训一版（`scripts/split_venue_holdout.py` 可做切分与留出）；
3. 用留出的后段帧验证，达到 mAP@0.5 ≥ 0.7 再上线。

实测收益：并入新场地 72 帧后，同场地后段 mAP 从 **0.267 提升到 0.746**（时间切分，偏乐观）。

⚠️ **预填时绝对不要用待评估的那个模型**，否则标注等于模型输出，指标会虚高
（本项目踩过：s04 有 80.8% 的框与模型输出 IoU≥0.9，算出假的 0.847）。
用 `scripts/label_independence.py` 先核对，IoU 中位数应明显低于 0.9。

## 8. 已知限制

1. **预标注只是省手工**：球拍静止、严重运动模糊、背景圆物多时会漏或错，人工必须复核。
2. **侧别是物理半台**：换边后同一支球拍会变成另一侧，这是交接包 §3 的约定，不是 bug；压网线判 `UNKNOWN`。
3. **权重未校准**：`HitWeights` 只是实验基线，必须在验证集校准、按场次隔离的测试集上冻结。
4. **跟踪按帧计龄**：抽帧间隔大时请同时设置 `TrackerConfig.max_age_s`，否则会频繁断轨。
5. **不判分**：球拍证据不能单独确认落台、得分、擦边或擦网；没检到球拍也不等于没击球。
6. **击球时刻召回未严格测量**：缺击球区间标注，目前只能用帧级召回推算（同场地约 0.6、
   新场地未标定约 0.3、标定后约 0.75）。

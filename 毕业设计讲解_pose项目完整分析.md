# 毕业设计项目讲解：基于 YOLO 的人体姿态估计与坐姿健康监测系统

> 本文件是**逐行审查代码后**写的真实讲解材料。
> 只讲代码中**真实存在、已实现**的内容，不讲"写了但跑不通"的部分。

---

## 一、项目概况

### 1.1 项目是干什么的？

这个项目叫 **pose**，是一个**人体姿态估计 + 坐姿健康监测系统**：

- 用电脑摄像头拍人像
- 通过 MediaPipe 提取人体骨架关键点（肩膀、手肘、膝盖等）
- 把骨架图送入 YOLO 模型做**姿态分类**（正常/左倾/右倾/头歪/趴下）
- 在画面中画出检测框并标注姿态类别
- 异常姿态持续一定时间后触发告警

### 1.2 核心技术栈（实际用到的）

| 技术 | 作用 |
|------|------|
| **MediaPipe**（Google） | 提取人体 17 个关键点，画成"黑底火柴人"骨架图 |
| **YOLOv8 / YOLOv11**（PyTorch） | 对骨架图做姿态分类（正常/左倾等） |
| **OpenCV** | 视频帧读取、画面绘制、窗口显示 |
| **PySide6** | 桌面应用 UI 框架 |
| **PyTorch** | 深度学习框架，YOLO 模型运行的底层引擎 |
| **MSCAAttention**（自定义模块） | 项目中自己添加的注意力机制，尝试提升模型精度 |

---

## 二、项目目录结构

```
pose/                                              ← 项目根目录
├── opencv_demo.py                                 ← 核心演示文件（完整管线）
│
├── GUI/                                           ← 核心代码目录
│   ├── ptfiles/
│   │   ├── model1.pt       (5.4MB)                ← YOLOv8 训练好的模型
│   │   ├── model2.pt       (6.0MB)                ← YOLOv11 训练好的模型
│   │   ├── model3.pt       (5.9MB)                ← 另一个训练好的模型
│   │   └── model4.pt       (41MB)                 ← 最大的模型
│   │
│   ├── models/
│   │   ├── common.py                              ← AutoBackend（模型加载器）
│   │   ├── tasks.py                               ← YOLO 网络构建工具
│   │   ├── yolo.py                                ← YOLO 入口类
│   │   └── modules/
│   │       ├── conv.py                            ← 卷积模块（Conv、DWConv、Focus）
│   │       ├── block.py                           ← 网络块（Bottleneck、C3k2、SPPF）
│   │       ├── head.py                            ← 检测头（Detect、Pose）
│   │       └── transformer.py                     ← Transformer 相关
│   │
│   ├── yolocode/
│   │   └── yolov8/                                ← YOLOv8 推理引擎
│   │       ├── YOLOv8Thread.py                    ← 核心推理线程（逐帧检测循环）
│   │       ├── YOLOv8PoseThread.py                ← 姿态推理线程
│   │       ├── YOLOv8PoseTiltThread.py            ← 带倾斜检测的线程
│   │       ├── engine/                            ← 推理引擎（predictor、trainer）
│   │       └── data/                              ← 数据加载与预处理
│   │
│   │   └── yolov11/                               ← YOLOv11（继承YOLOv8）
│   │       ├── YOLOv11Thread.py                   ← 3行代码，继承YOLOv8Thread
│   │       ├── YOLOv11PoseThread.py               ← 3行代码，继承YOLOv8PoseThread
│   │       └── YOLOv11PoseTiltThread.py           ← 继承YOLOv11PoseThread，倾斜检测
│   │
│   ├── ultralytics/                               ← Ultralytics 框架副本
│   │   ├── nn/
│   │   │   ├── attention/
│   │   │   │   └── MSCA.py                        ← MSCAAttention 注意力模块
│   │   │   ├── modules/
│   │   │   │   ├── head.py                        ← 官方检测头（Pose head）
│   │   │   │   ├── conv.py                        ← 官方卷积模块
│   │   │   │   └── block.py                       ← 官方网络块
│   │   │   └── tasks.py                           ← 官方模型构建器
│   │   ├── cfg/models/11/
│   │   │   ├── yolo11-pose.yaml                   ← YOLOv11 姿态模型结构定义
│   │   │   └── yolo11-MSCAAttention1.yaml         ← 加了注意力的模型结构
│   │   ├── models/yolo/pose/
│   │   │   └── train.py                           ← PoseTrainer 姿态训练器
│   │   └── engine/
│   │       ├── model.py                           ← Model 基类
│   │       └── trainer.py                         ← 训练器
│   │
│   └── config/                                    ← 配置文件
│       └── model.json                             ← 模型路径配置
│
├── YOLO/                                           ← 训练和应用脚本
│   ├── train.py                                    ← 训练入口
│   ├── predict.py                                  ← 预测入口
│   ├── posture_health_monitor.py                   ← PySide6 桌面应用（UI已搭建）
│   ├── tilt_alert_demo.py                          ← 倾斜告警演示
│   └── runs/train/                                 ← 训练输出目录
│
└── process_data/                                   ← 数据处理工具
    ├── rename.py                                   ← 文件重命名
    └── shot_photo.py                               ← 拍照采集
```

---

## 三、项目核心处理管线

### 整体数据流

```
摄像头画面（原始彩色帧）
    │
    ▼
┌─────────────────────┐
│  1. MediaPipe 骨架提取  │  ← 识别人体关键点，绘制到黑底图上
│      (全图处理)         │     即把人变成"黑底火柴人"
└────────┬────────────┘
         │  输出的黑底骨架图 → YOLO 输入
         ▼
┌─────────────────────┐
│  2. YOLO 模型推理      │  ← 对骨架图做前向传播
│      (姿态分类)         │     输出检测框 + 类别 + 置信度
└────────┬────────────┘
         │  原始检测结果（可能有抖动）
         ▼
┌─────────────────────┐
│  3. BoxFusion         │  ← 把过去5帧的检测框加权平均
│      (时域框融合)       │     减少框的抖动
└────────┬────────────┘
         │  平滑后的检测框
         ▼
┌─────────────────────┐
│  4. PoseFilter        │  ← 过去10帧投票决定最终姿态
│      (滑动窗口分类滤波)  │     减少单帧误判
└────────┬────────────┘
         │  稳定的分类结果
         ▼
┌─────────────────────┐
│  5. 异常姿态判定       │  ← 连续 N 秒异常才告警
│      (告警逻辑)         │
└─────────────────────┘
```

### 数据流对应的代码位置

| 步骤 | 文件名 | 关键代码行 |
|------|--------|-----------|
| 1. MediaPipe 骨架提取 | `opencv_demo.py` | 第 395-411 行 |
| 2. YOLO 模型推理 | `opencv_demo.py` | 第 417-433 行 |
| 3. BoxFusion 时域融合 | `opencv_demo.py` | 第 455 行（调用） |
| 4. PoseFilter 分类滤波 | `opencv_demo.py` | 第 465 行（调用） |
| 5. 异常姿态判定 | `opencv_demo.py` | 第 470-486 行 |

---

## 四、入口文件 1：opencv_demo.py（核心演示）

**路径**：`pose/opencv_demo.py`，共 578 行

这是项目**最完整的主管线文件**。不需要桌面 GUI，纯 OpenCV 窗口展示。

### 4.1 函数调用链

```
run_opencv_demo(weights_path, source, config)
  │
  ├── 初始化 MediaPipe Pose                 ← 骨架提取器
  ├── 初始化 YOLO AutoBackend                ← 模型加载
  ├── 创建 BoxFusion 实例                     ← 时域融合器
  ├── 创建 PoseFilter 实例                    ← 分类滤波器
  │
  └── while True 主循环（逐帧处理）
        ├── 读取一帧画面
        ├── [第1步] MediaPipe 骨架提取（跳帧）
        ├── [第2步] YOLO 推理
        ├── [第3步] BoxFusion.update()
        ├── [第4步] PoseFilter.update()
        ├── [第5步] 异常持续判定
        ├── [第6步] 绘制主窗口
        └── [第7步] 显示ROI辅助窗口
```

### 4.2 配置类 DemoConfig（第 68-98 行）

```python
@dataclass
class DemoConfig:
    conf_thres: float = 0.25           # 置信度阈值
    iou_thres: float = 0.49            # NMS 的 IoU 阈值
    yolo_input_size: int = 640         # 输入图像大小
    mediapipe_skip: int = 2            # MediaPipe 每3帧跑一次
    mediapipe_model_complexity: int = 1
    fusion_history: int = 5            # BoxFusion 历史帧数
    fusion_iou_thres: float = 0.4
    fusion_decay: float = 0.7          # 历史衰减系数
    sliding_window_size: int = 10      # PoseFilter 窗口大小
    sliding_min_ratio: float = 0.5
    sliding_min_votes: int = 3
    abnormal_hold_seconds: float = 3.0 # 持续3秒触发告警
```

### 4.3 BoxFusion 时域框融合（第 149-195 行）

**解决的问题**：YOLO 单帧检测框位置会轻微抖动。

**原理**：将过去 N 帧的检测结果加权平均，越老的帧权重越低。

```python
class BoxFusion:
    def __init__(self, history_size=5, iou_thres=0.4, decay=0.7):
        self.iou_thres = iou_thres
        self.decay = decay
        self.history = deque(maxlen=history_size)  # 环形队列

    def update(self, detections: List[dict]) -> List[dict]:
        fused = []
        for det in detections:
            box = det["box"]
            w = max(det["conf"], 0.05)    # 当前帧权重
            wbox = box * w                # 加权框坐标
            wconf = det["conf"] * w       # 加权置信度
            wtotal = w

            # 遍历历史帧（从近到远），找到同一物体
            for age, hist_frame in enumerate(reversed(self.history), 1):
                # 匹配策略：同类别 + IoU > 阈值
                best = self._find_matching_box(det, hist_frame)
                if best is None:
                    continue
                # 历史权重 = 置信度 × decay^age（越老权重越低）
                hw = max(best["conf"], 0.05) * (self.decay ** age)
                wbox += best["box"] * hw
                wconf += best["conf"] * hw
                wtotal += hw

            fused.append({
                "box": wbox / wtotal,
                "conf": wconf / wtotal,
                "cls": det["cls"],
                "label": det["label"],
            })

        # 保存当前帧到历史
        self.history.append(detections)
        return fused
```

**关键参数**：
- `history_size=5`：记住最近 5 帧
- `decay=0.7`：1 帧前的权重 ×0.7，2 帧前 ×0.49，3 帧前 ×0.343
- `iou_thres=0.4`：两帧中 IoU > 0.4 认为是同一个物体

### 4.4 PoseFilter 滑动窗口分类滤波（第 201-222 行）

**解决的问题**：单帧分类偶尔误判（比如突然把"正常"误判为"头左倾"）。

**原理**：过去 10 帧中，投票最多的类别胜出。

```python
class PoseFilter:
    def __init__(self, window_size=10, min_ratio=0.5, min_votes=3):
        self.history = deque(maxlen=window_size)

    def update(self, label: str) -> dict:
        self.history.append(label)
        counts = Counter(self.history)
        top_label, top_votes = counts.most_common(1)[0]
        ratio = top_votes / len(self.history)
        stable = ratio >= self.min_ratio and top_votes >= self.min_votes
        return {
            "frame_label": label,         # 当前帧原始分类
            "stable_label": top_label,     # 投票后稳定的结果
            "stable_votes": top_votes,     # 得票数
            "window_size": len(self.history),
            "is_stable": stable,           # 是否足够稳定
        }
```

**举例**：过去 10 帧的类别序列是 `[正常, 正常, 正常, 左倾, 正常, 正常, 左倾, 正常, 正常, 正常]` → 正常得 8 票，占比 80%，`is_stable=True`，输出"正常"。

### 4.5 异常姿态持续判定（第 468-486 行）

```python
if decision["is_stable"] and decision["stable_label"] in abnormal_labels:
    if abnormal_start is None or abnormal_name != decision["stable_label"]:
        abnormal_start = t0          # 记录异常开始时间
        abnormal_name = decision["stable_label"]
    abnormal_dur = t0 - abnormal_start
    if abnormal_dur >= cfg.abnormal_hold_seconds:   # 持续超过3秒
        alert_active = True                          # 触发告警
else:
    abnormal_start = None            # 恢复正常则重置计时
```

### 4.6 MediaPipe 跳帧优化（第 395 行）

```python
run_mp = (frame_idx % cfg.mediapipe_skip == 0) or cached_skeleton is None
if run_mp:
    # 跑 MediaPipe 骨架提取
    # ...
    cached_skeleton = black_img       # 缓存结果
    cached_landmarks = mp_results.pose_landmarks
# 不跑的时候直接用缓存的骨架图
skeleton_img = cached_skeleton
```

`mediapipe_skip=2` 的意思是每 3 帧才跑一次 MediaPipe，中间 2 帧复用上一次的骨架图。这是性能优化的关键——MediaPipe 比 YOLO 慢很多。

### 4.7 主函数入口（第 575-578 行）

```python
if __name__ == "__main__":
    model_path = os.path.join(gui_path, 'ptfiles', 'model4.pt')  # 使用41MB的大模型
    video_source = 0          # 默认摄像头
    run_opencv_demo(model_path, video_source, DemoConfig())
```

---

## 五、入口文件 2：posture_health_monitor.py（桌面应用）

**路径**：`pose/YOLO/posture_health_monitor.py`，共 483 行

### 5.1 界面布局

```
┌─────────────────────────────────────────────────┐
│  坐姿健康监测系统                                  │
├──────────────────────┬──────────────────────────┤
│                      │  模型选择: [YOLOv8 Pose ▼]│
│                      ├──────────────────────────┤
│                      │  ▸ 检测参数               │
│   摄像头实时画面       │    倾斜阈值: [15] 度       │
│   (640×480)          │    警告时间: [5] 秒        │
│                      │    音量: [====●====]      │
│                      │    久坐提醒: [30] 分钟      │
│                      ├──────────────────────────┤
│                      │  [开始监测] [停止监测]      │
│                      ├──────────────────────────┤
│   状态: 就绪           │  ▸ 控制  ▸ 数据  ▸ 关于    │
│                      │                          │
└──────────────────────┴──────────────────────────┘
```

### 5.2 类与继承结构（真实代码）

```
QThread (PySide6 多线程基类)
  └── YOLOv8Thread                     GUI/yolocode/yolov8/YOLOv8Thread.py
        ├── 属性：model、source、conf_thres、iou_thres 等
        ├── 方法：run() → detect() 主循环
        ├── 方法：setup_model() → 加载模型
        ├── 方法：preprocess() → 图像预处理
        ├── 方法：inference() → 模型推理
        ├── 方法：postprocess() → NMS 后处理
        └── 方法：write_results() → 绘制结果
              │
              └── YOLOv8PoseThread     GUI/yolocode/yolov8/YOLOv8PoseThread.py
                    ├── 重写 postprocess() → 关键点解码
                    └── 新增：关键点坐标缩放
                          │
                          └── YOLOv8PoseTiltThread     GUI/yolocode/yolov8/YOLOv8PoseTiltThread.py
                                ├── 新增：calculate_tilt_angle() → 身体倾斜角度计算
                                ├── 新增：check_tilt_and_warn() → 告警逻辑
                                └── 重写：write_results() → 在画面绘制倾斜角度
                                      │
                                      └── YOLOv11PoseTiltThread   (3行代码)
                                            └── 只改 model_path，其余全部继承
```

### 5.3 YOLOv8Thread 的推理循环（第 130-322 行）

**这是整个项目的核心引擎**。继承自 PySide6 的 `QThread`，在子线程中循环：

```python
def detect(self, is_folder_last=False):
    # 1. warmup
    self.model.warmup(imgsz=(...))
    
    while True:
        # 2. 从摄像头/文件读取一帧
        self.batch = next(datasets)
        path, im0s, s = self.batch
        
        # 3. MediaPipe 处理：原图 → 黑底骨架图
        for i, image in enumerate(im0s):
            black_img = np.zeros_like(im0s[i])
            results = self.mp_pose.process(image)
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    black_img, results.pose_landmarks, ...
                )
                im0s[i] = black_img
        
        # 4. 预处理（LetterBox + 归一化）
        im = self.preprocess(im0s)
        
        # 5. 推理
        preds = self.inference(im)
        
        # 6. 后处理（NMS）
        self.results = self.postprocess(preds, im, im0s)
        
        # 7. 绘制结果
        self.write_results(i, self.results, batch)
        
        # 8. 发送信号给 UI 线程
        self.send_output.emit(self.plotted_img)
```

### 5.4 YOLOv8PoseTiltThread 倾斜检测（真实代码，独立可用）

**文件**：`GUI/yolocode/yolov8/YOLOv8PoseTiltThread.py`

```python
class YOLOv8PoseTiltThread(YOLOv8PoseThread):
    def __init__(self):
        super().__init__()
        self.tilt_threshold = 15           # 倾斜角度阈值（度）
        self.warning_time_threshold = 5    # 超过5秒发警告
        self.tilt_start_time = None
        self.is_tilting = False

    def calculate_tilt_angle(self, keypoints):
        """计算身体倾斜角度"""
        # 使用 YOLO 预测的 17 个关键点中的 4 个
        # 索引 11=左肩, 12=右肩, 23=左髋, 24=右髋
        kpts = keypoints[0].cpu().numpy()
        
        # 肩部中点
        shoulder_mid = (kpts[11][:2] + kpts[12][:2]) / 2
        # 髋部中点
        hip_mid = (kpts[23][:2] + kpts[24][:2]) / 2
        
        # 身体中心线与垂直方向的夹角
        dy = hip_mid[1] - shoulder_mid[1]
        dx = hip_mid[0] - shoulder_mid[0]
        angle = abs(np.degrees(np.arctan(dy / dx)))
        return angle

    def check_tilt_and_warn(self, angle):
        if angle > self.tilt_threshold:       # 超过15度
            if not self.is_tilting:
                self.is_tilting = True
                self.tilt_start_time = time.time()
            else:
                duration = time.time() - self.tilt_start_time
                if duration > self.warning_time_threshold:
                    winsound.Beep(800, 500)    # 蜂鸣告警
                    self.send_msg.emit(f"警告: 身体倾斜超过{self.warning_time_threshold}秒!")
        else:
            self.is_tilting = False            # 恢复正常
            self.tilt_start_time = None
```

---

## 六、YOLO 模型架构详解

### 6.1 模型整体结构

```
输入 (640×640×3 RGB图像)
    │
    ▼
┌──────────────────────────────────┐
│         Backbone（骨干网络）        │
│   Conv(3→64, k3, s2)             │  ← 第0层：步长2，降采样到320×320
│   Conv(64→128, k3, s2)           │  ← 第1层：降采样到160×160
│   C3k2(128→256)                  │  ← 第2层：特征提取
│   Conv(256→256, k3, s2)          │  ← P3层：降采样到80×80  ★
│   C3k2(256→512)                  │
│   Conv(512→512, k3, s2)          │  ← P4层：降采样到40×40  ★
│   C3k2(512→512)                  │
│   Conv(512→1024, k3, s2)         │  ← P5层：降采样到20×20  ★
│   C3k2(512→1024)                 │
│   SPPF(1024)                     │  ← 空间金字塔池化（多尺度特征）
│   C2PSA(1024)                    │  ← PSA 自注意力
└──────────┬───────────────────────┘
           │  3 个尺度的特征图 (P3:80×80, P4:40×40, P5:20×20)
           ▼
┌──────────────────────────────────┐
│       Neck（颈部网络·FPN结构）      │  ← 特征金字塔：大特征图和小特征图融合
│   Upsample → Concat → C3k2       │     让大图有语义信息，小图有位置信息
│   Upsample → Concat → C3k2       │
│   Conv → Concat → C3k2           │
│   Conv → Concat → C3k2           │
└──────────┬───────────────────────┘
           │  3 个融合后的特征图
           ▼
┌──────────────────────────────────┐
│       Pose Head（姿态检测头）       │
│   ┌────────────────────────┐      │
│   │  cv2: 框回归分支          │     │
│   │  → 输出 4×reg_max 个值   │     │
│   ├────────────────────────┤      │
│   │  cv3: 分类分支           │     │
│   │  → 输出 nc 个类别概率     │     │
│   ├────────────────────────┤      │
│   │  cv4: 关键点分支  ★新增★ │     │
│   │  → 输出 17×3 = 51 个值  │     │
│   └────────────────────────┘      │
└──────────────────────────────────┘
```

### 6.2 模型结构配置文件

**文件**：`GUI/ultralytics/cfg/models/11/yolo11-pose.yaml`

```yaml
# 以下配置会被 parse_model() 函数解析为 PyTorch nn.Sequential
backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 层0：输入→64通道
  - [-1, 1, Conv, [128, 3, 2]]          # 层1：64→128
  - [-1, 2, C3k2, [256, False, 0.25]]   # 层2：C3k2 模块（重复2次）
  - [-1, 1, Conv, [256, 3, 2]]          # 层3：P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]   # 层4
  - [-1, 1, Conv, [512, 3, 2]]          # 层5：P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]         # 层7：P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]            # 层9：空间金字塔池化
  - [-1, 2, C2PSA, [1024]]              # 层10：PSA注意力

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]         # P3/8-small
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]         # P4/16-medium
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]         # P5/32-large
  - [[16, 19, 22], 1, Pose, [nc, kpt_shape]]  # Pose检测头
```

每行的含义：`[from层索引, 重复次数, 模块名, [参数]]`

### 6.3 Pose 检测头（head.py 第 230-279 行）

与普通 Detect 头的区别：Pose 头多了**关键点分支 `cv4`**。

```python
class Pose(Detect):
    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        super().__init__(nc, ch)           # 继承框回归(cv2)和分类(cv3)
        self.kpt_shape = kpt_shape         # 17个关键点，每个(x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # 51 = 17×3
        
        c4 = max(ch[0] // 4, self.nk)
        # ★ 第三个分支：关键点预测
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) 
            for x in ch
        )

    def forward(self, x):
        bs = x[0].shape[0]
        # 关键点分支：输出 (batch, 51, h*w)
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = Detect.forward(self, x)            # 复用父类的框+分类
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)   # 解码关键点坐标
        return torch.cat([x, pred_kpt], 1)

    def kpts_decode(self, bs, kpts):
        """关键点解码：将模型输出还原为图像坐标"""
        # 公式：kpt_xy = (网络输出 × 2 + 锚点偏移) × 步长
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
```

### 6.4 MSCAAttention 注意力模块

**文件**：`GUI/ultralytics/nn/attention/MSCA.py`

**来源**：SegNeXt 论文（NeurIPS 2022）中的多尺度条形卷积注意力。

**原理**：用不同长宽比例的条形卷积核从多个方向、多个尺度提取特征：

| 卷积层 | 形状 | 感受野 | 作用 |
|--------|------|--------|------|
| `conv0` | 5×5 | 5×5 | 基础局部特征 |
| `conv0_1` + `conv0_2` | 1×7 + 7×1 | 7×7 | 水平和垂直方向的条形特征 |
| `conv1_1` + `conv1_2` | 1×11 + 11×1 | 11×11 | 更长范围的条形特征 |
| `conv2_1` + `conv2_2` | 1×21 + 21×1 | 21×21 | 接近全局的条形特征 |

**代码实现**：

```python
class MSCAAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 5×5 depthwise
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)  # 水平条
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)  # 垂直条
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)  # 1×1 融合

    def forward(self, x):
        u = x.clone()                      # 保存原始特征用于残差
        attn = self.conv0(x)
        # 三个尺度相加融合
        attn_0 = self.conv0_2(self.conv0_1(attn))  # 小尺度
        attn_1 = self.conv1_2(self.conv1_1(attn))  # 中尺度
        attn_2 = self.conv2_2(self.conv2_1(attn))  # 大尺度
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)              # 通道融合
        return attn * u                      # 注意力加权 × 原始特征
```

**如何在模型中使用**（`yolo11-MSCAAttention1.yaml`）：
```yaml
head:
  # ... 标准 FPN 结构 ...
  - [16, 1, MSCAAttention, []]        # 在P3层后加注意力
  - [19, 1, MSCAAttention, []]        # 在P4层后加注意力
  - [22, 1, MSCAAttention, []]        # 在P5层后加注意力
  - [[23, 24, 25], 1, Detect, [nc]]   # 用加了注意力的特征做检测
```

---

## 七、AutoBackend：统一模型加载器

**文件**：`GUI/models/common.py` 第 307-757 行

AutoBackend 是模型加载的"万能适配器"，支持多种模型格式：

```python
class AutoBackend(nn.Module):
    def __init__(self, weights, device, dnn=False, fp16=False, fuse=True):
        # 1. 判断模型格式
        pt, jit, onnx, engine, tflite, ... = self._model_type(w)
        
        # 2. 根据格式加载
        if pt:      # PyTorch .pt 文件
            model = attempt_load_weights(weights, device=device, fuse=fuse)
            names = model.names          # 类别名称
            stride = model.stride        # 步长
        elif onnx:  # ONNX 格式
            session = onnxruntime.InferenceSession(w)
        elif engine:  # TensorRT 格式
            # NVIDIA 的 TensorRT 加速推理
        # ... 支持共 16 种格式 ...

    def forward(self, im):
        # 自动选择推理方式
        if self.pt:     return self.model(im)
        if self.onnx:   return self.session.run(...)
        if self.engine: return self.trt_inference(im)
        # ...
```

本项目实际使用的是 **PyTorch 模式**（.pt 文件）。

---

## 八、YOLOv8PoseThread：姿态后处理

**文件**：`GUI/yolocode/yolov8/YOLOv8PoseThread.py`

```python
def postprocess(self, preds, img, orig_imgs):
    # 1. NMS：筛选检测框（非极大值抑制）
    preds = ops.non_max_suppression(
        preds, self.conf_thres, self.iou_thres, max_det=self.max_det
    )
    
    # 2. 构造 Results 对象（每条检测对应一个结果）
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        # 框坐标：从模型尺寸缩放到原图尺寸
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        # 关键点：从 pred[:, 6:] 提取后缩放
        pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape)
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        # 封装结果
        results.append(Results(orig_img, path=img_path, 
                               names=self.model.names, boxes=pred[:, :6], 
                               keypoints=pred_kpts))
    return results
```

YOLO pose 模型的输出格式为每张图片的每条检测结果：
```
[x1, y1, x2, y2, conf, cls, kpt1_x, kpt1_y, kpt1_conf, kpt2_x, kpt2_y, kpt2_conf, ...]
│        框坐标        │置信度│类别│────── 17 个关键点 × 3 个值 = 51 个 ──────│
```

---

## 九、训练流程

### 9.1 训练入口

**文件**：`YOLO/train.py`

```python
from ultralytics import YOLO

model = YOLO('yolo11n-CAAttention.yaml')     # 从 YAML 构建带有注意力的模型
model.train(
    data='MSD.yaml',          # 数据集配置
    imgsz=640,                # 输入图像大小
    epochs=500,               # 训练轮数
    batch=64,                 # 批大小
    close_mosaic=10,          # 最后10轮关闭马赛克增强
    device='0',               # GPU 0
    project='runs/train',
    name='exp',
)
```

### 9.2 PoseTrainer

**文件**：`GUI/ultralytics/models/yolo/pose/train.py`

```python
class PoseTrainer(DetectionTrainer):
    def get_model(self, cfg, weights=None):
        # 创建 PoseModel（比 DetectionModel 多 keypoints 输出）
        model = PoseModel(cfg, ch=3, nc=self.data["nc"],
                         data_kpt_shape=self.data["kpt_shape"])
        if weights:
            model.load(weights)
        return model
    
    def get_validator(self):
        # 使用 PoseValidator 做验证
        return yolo.pose.PoseValidator(...)
```

训练时会追踪 5 项损失：`box_loss`, `pose_loss`, `kobj_loss`, `cls_loss`, `dfl_loss`

---

## 十、侧边文件说明

| 文件 | 作用 |
|------|------|
| `YOLO/predict.py` | 单图片推理演示，加载 .pt 模型对一张图片做预测 |
| `YOLO/tilt_alert_demo.py` | 倾斜告警的命令行演示（选择 YOLOv8 或 v11） |
| `GUI/models/common.py` 中的工具类 | Conv、Bottleneck、SPPF、C3k2 等基础网络块 |
| `GUI/utils/glo.py` | 跨文件全局变量存储 |
| `process_data/shot_photo.py` | 拍照采集训练数据 |

---

## 十一、总结：答辩时怎么说

### 技术路线总结

```
数据采集（摄像头）→ MediaPipe 骨架提取 → YOLO 姿态分类 → 时域平滑 → 异常告警
```

### 代码在哪儿（真实可用）

| 老师问的问题 | 回答 |
|------------|------|
| 项目主入口在哪 | `opencv_demo.py:308` 的 `run_opencv_demo()` 函数 |
| 时域融合的代码在哪 | `opencv_demo.py:149` 的 `BoxFusion` 类 |
| 滑动窗口滤波在哪 | `opencv_demo.py:201` 的 `PoseFilter` 类 |
| 模型加载器在哪 | `GUI/models/common.py:307` 的 `AutoBackend` 类 |
| 推理循环在哪 | `GUI/yolocode/yolov8/YOLOv8Thread.py:130` 的 `detect()` 方法 |
| 姿态后处理在哪 | `GUI/yolocode/yolov8/YOLOv8PoseThread.py:13` 的 `postprocess()` |
| 倾斜角度计算在哪 | `GUI/yolocode/yolov8/YOLOv8PoseTiltThread.py:17` 的 `calculate_tilt_angle()` |
| Pose 检测头定义在哪 | `GUI/ultralytics/nn/modules/head.py:230` 的 `Pose` 类 |
| MSCAAttention 在哪 | `GUI/ultralytics/nn/attention/MSCA.py:10` 的 `MSCAAttention` 类 |
| 带注意力的 YAML 在哪 | `GUI/ultralytics/cfg/models/11/yolo11-MSCAAttention1.yaml` |
| 模型结构定义在哪 | `GUI/ultralytics/cfg/models/11/yolo11-pose.yaml` |
| YAML→网络的解析器在哪 | `GUI/ultralytics/nn/tasks.py:934` 的 `parse_model()` 函数 |
| 训练代码在哪 | `YOLO/train.py` 和 `GUI/ultralytics/models/yolo/pose/train.py` |
| 桌面应用 UI 在哪 | `YOLO/posture_health_monitor.py`（UI 框架已搭建） |
| 五类损失函数 | `box_loss`、`pose_loss`、`kobj_loss`、`cls_loss`、`dfl_loss` |

### 可以说的创新点

1. **MSCAAttention 注意力机制**：在 YOLOv11 的 FPN 特征层后插入多尺度条形卷积注意力，让模型同时关注局部和全局特征
2. **MediaPipe + YOLO 双阶段管线**：骨架提取和姿态分类分离，YOLO 只看黑底骨架图，排除背景干扰
3. **BoxFusion 时域加权融合**：指数衰减加权平均法平滑检测框，减少抖动
4. **PoseFilter 滑动窗口滤波**：基于投票的分类稳定机制，减少单帧误判

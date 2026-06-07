# =================================================================
# YOLOv11 OpenCV Demo - 完整功能版
#
# 功能:
# 1. MediaPipe 全图骨架提取 -> 黑底图 -> YOLO 姿态识别 (原始管线不变)
# 2. ROI 区域提取 + 独立窗口展示 (固定展示尺寸, 不影响推理)
# 3. YOLO 检测框的时域加权融合 (抗抖动)
# 4. 滑动窗口分类滤波 (减少误识别)
# 5. 异常姿态持续判定 + 告警提示
# 6. 帧率优化: MediaPipe 跳帧 + cudnn.benchmark + 降低输入分辨率
# =================================================================

import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

# --- 路径设置 ---
project_root = os.path.dirname(os.path.abspath(__file__))
gui_path = os.path.join(project_root, 'GUI')
sys.path.append(gui_path)

try:
    from utils import glo
    glo._init()
    glo.set_value('yoloname', 'yolov11')
except ImportError:
    pass

try:
    from models.common import AutoBackend
    from yolocode.yolov8.utils import ops
    from yolocode.yolov8.data.augment import LetterBox
except ImportError as e:
    print(f"错误：导入项目组件失败: {e}")
    sys.exit(1)


# =====================================================================
# 类别中文名映射 (英文类名 -> 中文描述)
# =====================================================================
CLASS_NAMES_CN = {
    "normal":             "正常",
    "body_left":          "身体左倾",
    "body_right":         "身体右倾",
    "left_support_head":  "头左倾",
    "right_support_head": "头右倾",
    "lying_down":         "趴下/躺下",
    "No Pose":            "未检测到姿态",
}


def to_cn(label: str) -> str:
    return CLASS_NAMES_CN.get(label, label)


# =====================================================================
# 配置
# =====================================================================
@dataclass
class DemoConfig:
    # YOLO
    conf_thres: float = 0.25
    iou_thres: float = 0.49
    yolo_input_size: int = 640

    # MediaPipe
    mediapipe_skip: int = 2
    mediapipe_model_complexity: int = 1
    mediapipe_min_det_conf: float = 0.5
    mediapipe_min_track_conf: float = 0.5

    # ROI / 骨架辅助窗口 (仅展示, 不影响推理)
    roi_padding: float = 0.25
    roi_display_size: Tuple[int, int] = (480, 480)
    show_aux_window: bool = True          # ROI + 骨架合并窗口

    # 时域加权融合
    fusion_history: int = 5
    fusion_iou_thres: float = 0.4
    fusion_decay: float = 0.7

    # 滑动窗口滤波
    sliding_window_size: int = 10
    sliding_min_ratio: float = 0.5
    sliding_min_votes: int = 3

    # 异常姿态告警 (除 normal 外均为异常)
    normal_class_names: Tuple[str, ...] = ("normal",)
    abnormal_hold_seconds: float = 3.0    # 持续多少秒触发告警


# =====================================================================
# 工具函数
# =====================================================================
def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def build_abnormal_set(names, cfg):
    """除 normal_class_names 中列出的类别外, 其余均为异常"""
    normal_set = {s.strip().lower() for s in cfg.normal_class_names}
    result = set()
    for name in (names.values() if isinstance(names, dict) else names):
        if name.strip().lower() not in normal_set:
            result.add(name)
    return result


def landmarks_bbox(landmarks, w, h, padding=0.25):
    xs = [lm.x * w for lm in landmarks.landmark if lm.visibility > 0.3]
    ys = [lm.y * h for lm in landmarks.landmark if lm.visibility > 0.3]
    if not xs:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(bw, bh) * (1 + padding) / 2
    return (
        max(0, int(cx - half)), max(0, int(cy - half)),
        min(w, int(cx + half)), min(h, int(cy + half)),
    )


def crop_and_resize(frame, box, target_size):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    return cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)


# =====================================================================
# 时域加权框融合
# =====================================================================
class BoxFusion:
    def __init__(self, history_size=5, iou_thres=0.4, decay=0.7):
        self.iou_thres = iou_thres
        self.decay = decay
        self.history = deque(maxlen=history_size)

    def reset(self):
        self.history.clear()

    def update(self, detections: List[dict]) -> List[dict]:
        fused = []
        for det in detections:
            box = det["box"].astype(np.float64)
            w = max(det["conf"], 0.05)
            wbox = box * w
            wconf = det["conf"] * w
            wtotal = w

            for age, hist_frame in enumerate(reversed(self.history), 1):
                best, best_iou = None, self.iou_thres
                for h in hist_frame:
                    if h["cls"] != det["cls"]:
                        continue
                    iou = compute_iou(box, h["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best = h
                if best is None:
                    continue
                hw = max(best["conf"], 0.05) * (self.decay ** age)
                wbox += best["box"].astype(np.float64) * hw
                wconf += best["conf"] * hw
                wtotal += hw

            fused.append({
                "box": (wbox / max(wtotal, 1e-9)).astype(np.float32),
                "conf": wconf / max(wtotal, 1e-9),
                "cls": det["cls"],
                "label": det["label"],
            })

        self.history.append([{
            "box": d["box"].copy(), "conf": d["conf"],
            "cls": d["cls"], "label": d["label"],
        } for d in detections])

        return fused


# =====================================================================
# 滑动窗口分类滤波
# =====================================================================
class PoseFilter:
    def __init__(self, window_size=10, min_ratio=0.5, min_votes=3):
        self.min_ratio = min_ratio
        self.min_votes = min_votes
        self.history = deque(maxlen=window_size)

    def reset(self):
        self.history.clear()

    def update(self, label: str) -> dict:
        self.history.append(label)
        counts = Counter(self.history)
        top_label, top_votes = counts.most_common(1)[0]
        ratio = top_votes / max(len(self.history), 1)
        stable = ratio >= self.min_ratio and top_votes >= self.min_votes
        return {
            "frame_label": label,
            "stable_label": top_label,
            "stable_votes": top_votes,
            "window_size": len(self.history),
            "is_stable": stable,
        }


# =====================================================================
# 绘图工具
# =====================================================================
def draw_box(img, box, label, conf, color, lw=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(th + 6, y1)
    cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
    cv2.putText(img, txt, (x1 + 4, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)


# PIL 中文渲染 (优先 msyh.ttc, 自动回退)
_pil_font_cache: dict = {}


def _get_pil_font(size: int):
    if size in _pil_font_cache:
        return _pil_font_cache[size]
    try:
        from PIL import ImageFont
        for path in [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]:
            try:
                f = ImageFont.truetype(path, size)
                _pil_font_cache[size] = f
                return f
            except Exception:
                continue
    except ImportError:
        pass
    return None


def put_text_cn(img, text: str, pos, font_size: int = 40,
                color=(0, 0, 255), bold: bool = True):
    """在 img 上渲染中文文字 (使用 PIL), 原地修改"""
    font = _get_pil_font(font_size)
    if font is not None:
        try:
            from PIL import Image, ImageDraw
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            r, g, b = color[2], color[1], color[0]
            if bold:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw.text((pos[0] + dx, pos[1] + dy), text,
                              font=font, fill=(0, 0, 0))
            draw.text(pos, text, font=font, fill=(r, g, b))
            img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
    # 回退：ASCII
    cv2.putText(img, text.encode('ascii', 'replace').decode(),
                pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_size / 36.0, color, 3)


def draw_alert_corner(img, cn_label: str, duration: float):
    """在主窗口左下角渲染大红字告警"""
    h, w = img.shape[:2]
    line1 = f"持续状态异常：{cn_label}"
    line2 = f"已持续 {duration:.1f}s"
    fs1, fs2 = 52, 36
    # 半透明背景区域
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - fs1 - fs2 - 30), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    put_text_cn(img, line1, (20, h - fs1 - fs2 - 14), font_size=fs1,
                color=(0, 0, 255))
    put_text_cn(img, line2, (24, h - fs2 - 8), font_size=fs2,
                color=(0, 80, 255))


# =====================================================================
# 主函数
# =====================================================================
def run_opencv_demo(weights_path, source=0, cfg=None):
    cfg = cfg or DemoConfig()

    if not os.path.exists(weights_path):
        print(f"错误：找不到模型文件 {weights_path}")
        return

    # --- MediaPipe ---
    mp_pose = mp.solutions.pose.Pose(
        model_complexity=cfg.mediapipe_model_complexity,
        min_detection_confidence=cfg.mediapipe_min_det_conf,
        min_tracking_confidence=cfg.mediapipe_min_track_conf,
    )
    mp_draw = mp.solutions.drawing_utils

    # --- YOLO ---
    print(f"\n正在加载模型: {os.path.basename(weights_path)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cv2.setUseOptimized(True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = AutoBackend(weights=weights_path, device=device, fuse=True)
    model.eval()
    names = model.names
    stride = model.stride
    fp16 = model.fp16
    letterbox = LetterBox(
        (cfg.yolo_input_size, cfg.yolo_input_size), auto=True, stride=stride
    )

    abnormal_labels = build_abnormal_set(names, cfg)

    # --- 时域模块 ---
    box_fusion = BoxFusion(cfg.fusion_history, cfg.fusion_iou_thres, cfg.fusion_decay)
    pose_filter = PoseFilter(cfg.sliding_window_size, cfg.sliding_min_ratio, cfg.sliding_min_votes)

    # --- 打印信息 ---
    print(f"使用设备: {device}")
    print(f"默认参数: conf={cfg.conf_thres}, iou={cfg.iou_thres}, imgsz={cfg.yolo_input_size}")
    print(f"MediaPipe 每 {cfg.mediapipe_skip} 帧更新, complexity={cfg.mediapipe_model_complexity}")
    print(f"融合历史: {cfg.fusion_history} 帧, 滑动窗口: {cfg.sliding_window_size} 帧")
    print(f"告警阈值: 持续 {cfg.abnormal_hold_seconds}s 触发")
    print(f"正常类别: {sorted(cfg.normal_class_names)}")
    print(f"异常类别: {sorted(abnormal_labels)}")
    print(f"模型全部类别: {names}")
    print("按键: q=退出, r=重置时域状态")
    print("-" * 40)

    # --- 视频源 ---
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"错误：无法打开视频源 {source}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        model.warmup(imgsz=(1, 3, cfg.yolo_input_size, cfg.yolo_input_size))
    except Exception:
        pass

    # --- 窗口 ---
    win_main = "YOLOv11-EQ Pose Demo"
    win_aux  = "Aux View (ROI + Skeleton)"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    if cfg.show_aux_window:
        cv2.namedWindow(win_aux, cv2.WINDOW_NORMAL)

    # --- 状态变量 ---
    fps = 0.0
    frame_idx = 0
    cached_skeleton = None
    cached_landmarks = None
    abnormal_start = None
    abnormal_name = None
    alert_latched = False

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        # =============================================================
        # 第 1 步: MediaPipe 骨架提取 (全图, 可跳帧)
        # =============================================================
        run_mp = (frame_idx % cfg.mediapipe_skip == 0) or cached_skeleton is None
        if run_mp:
            black_img = np.zeros_like(frame)
            mp_results = mp_pose.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            if mp_results.pose_landmarks:
                mp_draw.draw_landmarks(
                    black_img,
                    mp_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                )
                cached_skeleton = black_img
                cached_landmarks = mp_results.pose_landmarks
            else:
                cached_skeleton = black_img
                cached_landmarks = None
        skeleton_img = cached_skeleton

        # =============================================================
        # 第 2 步: YOLO 推理 (全图骨架 -> 姿态分类)
        # =============================================================
        img_t = letterbox(image=skeleton_img)
        img_t = np.stack([img_t])
        img_t = img_t[..., ::-1].transpose((0, 3, 1, 2))
        img_t = np.ascontiguousarray(img_t)
        img_t = torch.from_numpy(img_t).to(device)
        img_t = img_t.half() if fp16 else img_t.float()
        img_t /= 255.0

        with torch.no_grad():
            preds = model(img_t)

        results_nms = ops.non_max_suppression(
            preds,
            conf_thres=cfg.conf_thres,
            iou_thres=cfg.iou_thres,
            nc=len(names),
        )

        # --- 解析检测结果 ---
        raw_dets = []
        pred = results_nms[0]
        if len(pred) > 0:
            pred[:, :4] = ops.scale_boxes(
                img_t.shape[2:], pred[:, :4], frame.shape
            )
            for row in pred.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = row[:6]
                cls_id = int(cls_id)
                raw_dets.append({
                    "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "conf": float(conf),
                    "cls": cls_id,
                    "label": names[cls_id],
                })

        # =============================================================
        # 第 3 步: 时域加权框融合
        # =============================================================
        fused_dets = box_fusion.update(raw_dets)

        # =============================================================
        # 第 4 步: 滑动窗口分类滤波
        # =============================================================
        if fused_dets:
            top = max(fused_dets, key=lambda d: d["conf"])
            frame_label = top["label"]
        else:
            frame_label = "No Pose"
        decision = pose_filter.update(frame_label)

        # =============================================================
        # 第 5 步: 异常姿态持续判定
        # =============================================================
        abnormal_dur = 0.0
        alert_active = False
        if decision["is_stable"] and decision["stable_label"] in abnormal_labels:
            if abnormal_start is None or abnormal_name != decision["stable_label"]:
                abnormal_start = t0
                abnormal_name = decision["stable_label"]
                alert_latched = False
            abnormal_dur = t0 - abnormal_start
            if abnormal_dur >= cfg.abnormal_hold_seconds:
                alert_active = True
                if not alert_latched:
                    print(f"[ALERT] {abnormal_name} 持续 {abnormal_dur:.1f}s !")
                    alert_latched = True
        else:
            abnormal_start = None
            abnormal_name = None
            alert_latched = False

        # =============================================================
        # 第 6 步: 绘制主窗口
        # =============================================================
        annotated = frame.copy()
        for det in fused_dets:
            color = (0, 0, 255) if det["label"] in abnormal_labels else (0, 255, 0)
            draw_box(annotated, det["box"], det["label"], det["conf"], color)

        # FPS
        dt = max(time.perf_counter() - t0, 1e-6)
        ifps = 1.0 / dt
        fps = ifps if fps == 0 else fps * 0.85 + ifps * 0.15

        cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 200, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        counts = Counter(d["label"] for d in fused_dets)
        det_text = " | ".join(
            f"{k}:{v}" for k, v in counts.items()
        ) if counts else "No Pose Detected"
        cv2.putText(annotated, det_text, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if decision["is_stable"]:
            sw_text = (f"Stable: {decision['stable_label']} "
                       f"({decision['stable_votes']}/{decision['window_size']})")
        else:
            sw_text = (f"Filtering: {decision['frame_label']} "
                       f"({decision['stable_votes']}/{decision['window_size']})")
        cv2.putText(annotated, sw_text, (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if alert_active:
            draw_alert_corner(annotated, to_cn(abnormal_name), abnormal_dur)

        cv2.imshow(win_main, annotated)

        # =============================================================
        # 第 7 步: 辅助窗口 (左:ROI彩色  右:骨架黑底, 合并展示)
        # =============================================================
        if cfg.show_aux_window:
            th_aux, tw_aux = cfg.roi_display_size[1], cfg.roi_display_size[0]

            # 左半: ROI 彩色画面
            roi_box = None
            if cached_landmarks is not None:
                roi_box = landmarks_bbox(cached_landmarks, w, h, cfg.roi_padding)
            if roi_box is not None:
                roi_view = crop_and_resize(annotated, roi_box, cfg.roi_display_size)
                # 在主窗口也画出 ROI 框 (黄色)
                rx1, ry1, rx2, ry2 = roi_box
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)
            else:
                roi_view = cv2.resize(annotated, cfg.roi_display_size)

            # 右半: 骨架黑底图
            skel_view = cv2.resize(skeleton_img, cfg.roi_display_size)

            # 加标签 (中文)
            put_text_cn(roi_view, "ROI 区域", (8, 2), font_size=30,
                        color=(0, 200, 255))
            put_text_cn(skel_view, "骨架输入", (8, 2), font_size=30,
                        color=(0, 255, 180))

            aux_frame = np.hstack([roi_view, skel_view])
            cv2.imshow(win_aux, aux_frame)

        # --- 按键处理 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            box_fusion.reset()
            pose_filter.reset()
            cached_skeleton = None
            cached_landmarks = None
            abnormal_start = None
            alert_latched = False
            print("[INFO] 时域状态已重置")

        frame_idx += 1

    cap.release()
    mp_pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = os.path.join(gui_path, 'ptfiles', 'model4.pt')
    video_source = 0
    run_opencv_demo(model_path, video_source, DemoConfig())

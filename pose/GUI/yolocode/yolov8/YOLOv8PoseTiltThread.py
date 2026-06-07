from yolocode.yolov8.YOLOv8PoseThread import YOLOv8PoseThread
import numpy as np
import time
import winsound
import cv2


class YOLOv8PoseTiltThread(YOLOv8PoseThread):

    def __init__(self):
        super(YOLOv8PoseTiltThread, self).__init__()
        self.tilt_threshold = 15  # 倾斜角度阈值
        self.warning_time_threshold = 5  # 倾斜超过5秒发出警告
        self.tilt_start_time = None  # 开始倾斜的时间
        self.is_tilting = False  # 是否正在倾斜

    def calculate_tilt_angle(self, keypoints):
        """计算身体倾斜角度"""
        if len(keypoints) == 0:
            return 0
        
        # 获取关键点位
        # 0: 鼻子, 11: 左肩, 12: 右肩, 23: 左臀, 24: 右臀
        kpts = keypoints[0].cpu().numpy()
        
        # 检查关键点是否有效
        if kpts.size < 25 * 3:  # 至少需要25个关键点
            return 0
        
        # 计算肩部中点
        left_shoulder = kpts[11][:2]
        right_shoulder = kpts[12][:2]
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        
        # 计算臀部中点
        left_hip = kpts[23][:2]
        right_hip = kpts[24][:2]
        hip_mid = (left_hip + right_hip) / 2
        
        # 计算身体中心线的角度
        dy = hip_mid[1] - shoulder_mid[1]
        dx = hip_mid[0] - shoulder_mid[0]
        
        if dx == 0:
            return 0
        
        angle = np.degrees(np.arctan(dy / dx))
        return abs(angle)

    def check_tilt_and_warn(self, angle):
        """检查倾斜并发出警告"""
        if angle > self.tilt_threshold:
            if not self.is_tilting:
                self.is_tilting = True
                self.tilt_start_time = time.time()
            else:
                # 计算倾斜持续时间
                tilt_duration = time.time() - self.tilt_start_time
                if tilt_duration > self.warning_time_threshold:
                    # 发出警告声
                    winsound.Beep(800, 500)  # 800Hz, 500ms
                    self.send_msg.emit(f"警告: 身体倾斜超过{self.warning_time_threshold}秒!")
        else:
            # 重置状态
            self.is_tilting = False
            self.tilt_start_time = None

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.data_path = p
        result = results[idx]
        log_string += result.verbose()
        result = results[idx]

        result.orig_img = self.ori_img[idx]

        # 计算身体倾斜角度
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            angle = self.calculate_tilt_angle(result.keypoints)
            # 检查倾斜并发出警告
            self.check_tilt_and_warn(angle)
            # 在图像上显示倾斜角度
            cv2.putText(self.ori_img[idx], f'Tilt Angle: {angle:.1f}°', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add bbox to image
        plot_args = {
            "line_width": self.line_thickness,
            "boxes": True,
            "conf": True,
            "labels": True,
        }
        self.plotted_img = result.plot(**plot_args)
        return log_string

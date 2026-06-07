from GUI.yolocode.yolov8.YOLOv8PoseTiltThread import YOLOv8PoseTiltThread
from GUI.yolocode.yolov11.YOLOv11PoseTiltThread import YOLOv11PoseTiltThread
import cv2
import sys
import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, 
    QPushButton, QComboBox, QSlider, QSpinBox, QGroupBox, QTabWidget, 
    QTextEdit, QFileDialog, QMessageBox, QCheckBox, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, QDateTime
from PySide6.QtGui import QImage, QPixmap, QFont


class PostureHealthMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("坐姿健康监测系统")
        self.setGeometry(100, 100, 1000, 700)
        
        # 初始化数据存储
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.data_file = os.path.join(self.data_dir, f"{self.current_date}.json")
        self.load_data()
        
        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：视频显示区域
        left_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)
        
        # 状态信息
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)
        
        main_layout.addLayout(left_layout, 2)
        
        # 右侧：控制面板和数据区域
        right_layout = QVBoxLayout()
        
        # 标签页
        self.tab_widget = QTabWidget()
        
        # 控制标签页
        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8 Pose", "YOLOv11 Pose", "YOLOv8 + MediaPipe"])
        model_layout.addWidget(self.model_combo)
        control_layout.addWidget(model_group)
        
        # 检测参数
        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout(param_group)
        
        # 倾斜阈值
        tilt_layout = QHBoxLayout()
        tilt_layout.addWidget(QLabel("倾斜阈值:"))
        self.tilt_threshold = QSpinBox()
        self.tilt_threshold.setRange(5, 45)
        self.tilt_threshold.setValue(15)
        tilt_layout.addWidget(self.tilt_threshold)
        tilt_layout.addWidget(QLabel("度"))
        param_layout.addLayout(tilt_layout)
        
        # 警告时间
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("警告时间:"))
        self.warning_time = QSpinBox()
        self.warning_time.setRange(1, 30)
        self.warning_time.setValue(5)
        time_layout.addWidget(self.warning_time)
        time_layout.addWidget(QLabel("秒"))
        param_layout.addLayout(time_layout)
        
        # 音量控制
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        volume_layout.addWidget(self.volume_slider)
        param_layout.addLayout(volume_layout)
        
        # 久坐提醒
        sitting_layout = QHBoxLayout()
        sitting_layout.addWidget(QLabel("久坐提醒:"))
        self.sitting_time = QSpinBox()
        self.sitting_time.setRange(10, 120)
        self.sitting_time.setValue(30)
        sitting_layout.addWidget(self.sitting_time)
        sitting_layout.addWidget(QLabel("分钟"))
        param_layout.addLayout(sitting_layout)
        
        control_layout.addWidget(param_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始监测")
        self.stop_button = QPushButton("停止监测")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)
        
        self.tab_widget.addTab(control_tab, "控制")
        
        # 数据标签页
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # 今日数据
        today_group = QGroupBox("今日数据")
        today_layout = QVBoxLayout(today_group)
        self.today_data = QTextEdit()
        self.today_data.setReadOnly(True)
        today_layout.addWidget(self.today_data)
        data_layout.addWidget(today_group)
        
        # 报告按钮
        report_layout = QHBoxLayout()
        self.generate_report = QPushButton("生成报告")
        self.export_report = QPushButton("导出报告")
        report_layout.addWidget(self.generate_report)
        report_layout.addWidget(self.export_report)
        data_layout.addLayout(report_layout)
        
        self.tab_widget.addTab(data_tab, "数据")
        
        # 关于标签页
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setPlainText("坐姿健康监测系统\n\n"+
                               "功能：\n"+
                               "- 实时监测坐姿状态\n"+
                               "- 检测身体倾斜并发出警告\n"+
                               "- 统计坐姿健康数据\n"+
                               "- 生成健康报告\n\n"+
                               "隐私保护：\n"+
                               "- 所有数据本地存储\n"+
                               "- 不上传原始视频\n"+
                               "- 数据仅用于个人健康分析")
        about_layout.addWidget(about_text)
        self.tab_widget.addTab(about_tab, "关于")
        
        right_layout.addWidget(self.tab_widget)
        main_layout.addLayout(right_layout, 1)
        
        # 连接信号槽
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.generate_report.clicked.connect(self.generate_health_report)
        self.export_report.clicked.connect(self.export_health_report)
        
        # 初始化线程和定时器
        self.thread = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 监测状态
        self.is_monitoring = False
        self.start_time = None
        self.sitting_start_time = None
        self.bad_posture_count = 0
        self.total_posture_checks = 0
        
        # 更新今日数据显示
        self.update_today_data()
    
    def load_data(self):
        """加载今日数据"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "date": self.current_date,
                "total_time": 0,
                "bad_posture_count": 0,
                "total_posture_checks": 0,
                "sitting_periods": [],
                "bad_posture_types": {}
            }
    
    def save_data(self):
        """保存今日数据"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def update_today_data(self):
        """更新今日数据显示"""
        total_time = self.data.get("total_time", 0)
        bad_posture_count = self.data.get("bad_posture_count", 0)
        total_posture_checks = self.data.get("total_posture_checks", 0)
        sitting_periods = self.data.get("sitting_periods", [])
        
        bad_posture_rate = 0
        if total_posture_checks > 0:
            bad_posture_rate = (bad_posture_count / total_posture_checks) * 100
        
        data_text = f"今日学习时长: {total_time // 60}分{total_time % 60}秒\n"
        data_text += f"不良坐姿次数: {bad_posture_count}\n"
        data_text += f"坐姿检测次数: {total_posture_checks}\n"
        data_text += f"不良坐姿率: {bad_posture_rate:.1f}%\n"
        data_text += f"久坐次数: {len(sitting_periods)}\n"
        
        self.today_data.setPlainText(data_text)
    
    def start_monitoring(self):
        """开始监测"""
        # 选择模型类型
        model_type = self.model_combo.currentText()
        
        # 初始化线程
        if model_type == "YOLOv8 Pose":
            self.thread = YOLOv8PoseTiltThread()
            model_path = "GUI/ptfiles/model1.pt"
        elif model_type == "YOLOv11 Pose":
            self.thread = YOLOv11PoseTiltThread()
            model_path = "GUI/ptfiles/model2.pt"
        else:  # YOLOv8 + MediaPipe
            self.thread = YOLOv8PoseTiltThread()
            model_path = "GUI/ptfiles/model1.pt"
        
        # 设置线程参数
        self.thread.new_model_name = model_path
        self.thread.source = "0"  # 使用默认摄像头
        self.thread.stop_dtc = False
        self.thread.is_continue = True
        self.thread.tilt_threshold = self.tilt_threshold.value()
        self.thread.warning_time_threshold = self.warning_time.value()
        
        # 连接线程信号
        self.thread.send_output.connect(self.update_video_frame)
        self.thread.send_msg.connect(self.update_message)
        
        # 启动线程
        self.thread.start()
        
        # 启动定时器
        self.timer.start(30)
        
        # 更新状态
        self.is_monitoring = True
        self.start_time = datetime.datetime.now()
        self.sitting_start_time = datetime.datetime.now()
        
        # 更新按钮状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("监测中...")
    
    def stop_monitoring(self):
        """停止监测"""
        if self.thread:
            self.thread.stop_dtc = True
            self.thread.wait()
        
        self.timer.stop()
        
        # 更新数据
        if self.is_monitoring:
            end_time = datetime.datetime.now()
            duration = int((end_time - self.start_time).total_seconds())
            self.data["total_time"] += duration
            
            # 添加久坐记录
            sitting_duration = int((end_time - self.sitting_start_time).total_seconds())
            if sitting_duration > 60:  # 超过1分钟才记录
                self.data["sitting_periods"].append({
                    "start": self.sitting_start_time.strftime("%H:%M:%S"),
                    "end": end_time.strftime("%H:%M:%S"),
                    "duration": sitting_duration
                })
            
            self.save_data()
            self.update_today_data()
        
        # 更新状态
        self.is_monitoring = False
        self.status_label.setText("已停止")
        
        # 更新按钮状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def update_video_frame(self, frame):
        """更新视频帧"""
        # 转换OpenCV图像为Qt图像
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 显示图像
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
        
        # 检查久坐时间
        if self.is_monitoring:
            current_time = datetime.datetime.now()
            sitting_duration = int((current_time - self.sitting_start_time).total_seconds())
            if sitting_duration >= self.sitting_time.value() * 60:
                # 发出久坐提醒
                self.status_label.setText("久坐提醒: 请起身活动!")
                winsound.Beep(600, 1000)  # 600Hz, 1秒
                self.sitting_start_time = current_time
    
    def update_message(self, message):
        """更新消息"""
        print(f"消息: {message}")
        if "警告" in message:
            # 记录不良坐姿
            self.bad_posture_count += 1
            self.total_posture_checks += 1
            self.data["bad_posture_count"] += 1
            self.data["total_posture_checks"] += 1
            
            # 记录不良坐姿类型
            if "倾斜" in message:
                self.data["bad_posture_types"]["倾斜"] = self.data["bad_posture_types"].get("倾斜", 0) + 1
            
            self.save_data()
            self.update_today_data()
            
            # 更新状态标签
            self.status_label.setText(message)
    
    def update_frame(self):
        """更新帧"""
        pass
    
    def generate_health_report(self):
        """生成健康报告"""
        # 计算今日数据
        total_time = self.data.get("total_time", 0)
        bad_posture_count = self.data.get("bad_posture_count", 0)
        total_posture_checks = self.data.get("total_posture_checks", 0)
        sitting_periods = self.data.get("sitting_periods", [])
        bad_posture_types = self.data.get("bad_posture_types", {})
        
        bad_posture_rate = 0
        if total_posture_checks > 0:
            bad_posture_rate = (bad_posture_count / total_posture_checks) * 100
        
        average_sitting_time = 0
        if sitting_periods:
            total_sitting = sum(period["duration"] for period in sitting_periods)
            average_sitting_time = total_sitting / len(sitting_periods)
        
        # 生成报告
        report = f"# 坐姿健康报告\n\n"
        report += f"## 日期: {self.current_date}\n\n"
        report += f"### 基本数据\n"
        report += f"- 学习总时长: {total_time // 60}分{total_time % 60}秒\n"
        report += f"- 不良坐姿次数: {bad_posture_count}\n"
        report += f"- 坐姿检测次数: {total_posture_checks}\n"
        report += f"- 不良坐姿率: {bad_posture_rate:.1f}%\n"
        report += f"- 久坐次数: {len(sitting_periods)}\n"
        report += f"- 平均久坐时长: {average_sitting_time // 60}分{average_sitting_time % 60}秒\n\n"
        
        if bad_posture_types:
            report += f"### 不良坐姿类型分布\n"
            for posture_type, count in bad_posture_types.items():
                report += f"- {posture_type}: {count}次\n"
        
        # 健康建议
        report += "\n### 健康建议\n"
        if bad_posture_rate > 30:
            report += "- 注意保持正确坐姿，避免长时间倾斜\n"
        if len(sitting_periods) > 5:
            report += "- 减少久坐时间，增加起身活动频率\n"
        if average_sitting_time > 45 * 60:
            report += "- 建议每30分钟起身活动一次\n"
        
        # 显示报告
        report_window = QMainWindow()
        report_window.setWindowTitle("坐姿健康报告")
        report_window.setGeometry(200, 200, 600, 500)
        
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)
        
        report_text = QTextEdit()
        report_text.setReadOnly(True)
        report_text.setPlainText(report)
        report_layout.addWidget(report_text)
        
        report_window.setCentralWidget(report_widget)
        report_window.show()
    
    def export_health_report(self):
        """导出健康报告"""
        # 生成报告内容
        total_time = self.data.get("total_time", 0)
        bad_posture_count = self.data.get("bad_posture_count", 0)
        total_posture_checks = self.data.get("total_posture_checks", 0)
        sitting_periods = self.data.get("sitting_periods", [])
        bad_posture_types = self.data.get("bad_posture_types", {})
        
        bad_posture_rate = 0
        if total_posture_checks > 0:
            bad_posture_rate = (bad_posture_count / total_posture_checks) * 100
        
        average_sitting_time = 0
        if sitting_periods:
            total_sitting = sum(period["duration"] for period in sitting_periods)
            average_sitting_time = total_sitting / len(sitting_periods)
        
        # 生成报告
        report = f"坐姿健康报告\n"
        report += f"日期: {self.current_date}\n\n"
        report += f"基本数据\n"
        report += f"- 学习总时长: {total_time // 60}分{total_time % 60}秒\n"
        report += f"- 不良坐姿次数: {bad_posture_count}\n"
        report += f"- 坐姿检测次数: {total_posture_checks}\n"
        report += f"- 不良坐姿率: {bad_posture_rate:.1f}%\n"
        report += f"- 久坐次数: {len(sitting_periods)}\n"
        report += f"- 平均久坐时长: {average_sitting_time // 60}分{average_sitting_time % 60}秒\n\n"
        
        if bad_posture_types:
            report += f"不良坐姿类型分布\n"
            for posture_type, count in bad_posture_types.items():
                report += f"- {posture_type}: {count}次\n"
        
        # 健康建议
        report += "\n健康建议\n"
        if bad_posture_rate > 30:
            report += "- 注意保持正确坐姿，避免长时间倾斜\n"
        if len(sitting_periods) > 5:
            report += "- 减少久坐时间，增加起身活动频率\n"
        if average_sitting_time > 45 * 60:
            report += "- 建议每30分钟起身活动一次\n"
        
        # 保存文件
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", f"posture_report_{self.current_date}.txt", "文本文件 (*.txt)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            QMessageBox.information(self, "成功", "报告导出成功!")
    
    def closeEvent(self, event):
        """关闭窗口时停止监测"""
        self.stop_monitoring()
        event.accept()


if __name__ == "__main__":
    # 确保winsound模块可用
    try:
        import winsound
    except ImportError:
        # 如果在非Windows系统上，定义一个模拟的winsound模块
        class winsound:
            @staticmethod
            def Beep(frequency, duration):
                pass
    
    app = QApplication(sys.argv)
    monitor = PostureHealthMonitor()
    monitor.show()
    sys.exit(app.exec())

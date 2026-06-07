from GUI.yolocode.yolov8.YOLOv8PoseTiltThread import YOLOv8PoseTiltThread
from GUI.yolocode.yolov11.YOLOv11PoseTiltThread import YOLOv11PoseTiltThread
import cv2
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap


class TiltDetectionDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("身体倾斜检测演示")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # 创建控制按钮
        self.start_button = QPushButton("开始检测")
        self.stop_button = QPushButton("停止检测")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8 Pose", "YOLOv11 Pose"])
        
        # 添加控制按钮到布局
        layout.addWidget(self.model_combo)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        # 连接信号槽
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        
        # 初始化线程和定时器
        self.thread = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 初始化摄像头
        self.cap = None
        
        # 停止按钮初始状态为禁用
        self.stop_button.setEnabled(False)
    
    def start_detection(self):
        # 选择模型类型
        model_type = self.model_combo.currentText()
        
        # 初始化线程
        if model_type == "YOLOv8 Pose":
            self.thread = YOLOv8PoseTiltThread()
            model_path = "GUI/ptfiles/model1.pt"  # 替换为实际的YOLOv8姿态模型路径
        else:
            self.thread = YOLOv11PoseTiltThread()
            model_path = "GUI/ptfiles/model2.pt"  # 替换为实际的YOLOv11姿态模型路径
        
        # 设置线程参数
        self.thread.new_model_name = model_path
        self.thread.source = "0"  # 使用默认摄像头
        self.thread.stop_dtc = False
        self.thread.is_continue = True
        
        # 连接线程信号
        self.thread.send_output.connect(self.update_video_frame)
        self.thread.send_msg.connect(self.update_message)
        
        # 启动线程
        self.thread.start()
        
        # 启动定时器
        self.timer.start(30)
        
        # 更新按钮状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_detection(self):
        if self.thread:
            self.thread.stop_dtc = True
            self.thread.wait()
        
        self.timer.stop()
        
        # 更新按钮状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def update_video_frame(self, frame):
        # 转换OpenCV图像为Qt图像
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 显示图像
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
    
    def update_message(self, message):
        # 显示消息
        print(f"消息: {message}")
    
    def update_frame(self):
        # 这里可以添加额外的帧更新逻辑
        pass
    
    def closeEvent(self, event):
        # 关闭窗口时停止线程
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = TiltDetectionDemo()
    demo.show()
    sys.exit(app.exec())

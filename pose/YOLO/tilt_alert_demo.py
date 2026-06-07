import cv2
import sys
from GUI.yolocode.yolov8.YOLOv8PoseTiltThread import YOLOv8PoseTiltThread
from GUI.yolocode.yolov11.YOLOv11PoseTiltThread import YOLOv11PoseTiltThread


def main():
    # 选择模型类型
    print("请选择模型类型:")
    print("1. YOLOv8 Pose")
    print("2. YOLOv11 Pose")
    choice = input("输入选项 (1/2): ")
    
    # 初始化线程
    if choice == "1":
        thread = YOLOv8PoseTiltThread()
        model_path = "GUI/ptfiles/model1.pt"
    else:
        thread = YOLOv11PoseTiltThread()
        model_path = "GUI/ptfiles/model2.pt"
    
    # 设置线程参数
    thread.new_model_name = model_path
    thread.source = "0"  # 使用默认摄像头
    thread.stop_dtc = False
    thread.is_continue = True
    
    # 设置倾斜检测参数
    thread.tilt_threshold = 15  # 倾斜角度阈值
    thread.warning_time_threshold = 3  # 倾斜超过3秒发出警告
    
    # 启动线程
    thread.start()
    
    print("身体倾斜检测已启动")
    print("当身体倾斜超过15度且持续3秒以上时，将发出警报")
    print("按 'q' 键退出")
    
    # 显示视频流
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 显示帧
        cv2.imshow('身体倾斜检测', frame)
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 停止线程
    thread.stop_dtc = True
    thread.wait()
    
    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

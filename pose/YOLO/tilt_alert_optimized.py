import cv2
import numpy as np
import time
import winsound
from ultralytics import YOLO


def calculate_tilt_angle(keypoints):
    """计算身体倾斜角度"""
    if len(keypoints) == 0:
        return 0
    
    # 获取关键点位
    # COCO关键点索引: 5-左肩, 6-右肩, 11-左臀, 12-右臀
    kpts = keypoints[0].cpu().numpy()
    
    # 检查关键点是否有效（COCO有17个关键点，每个关键点有x,y,confidence）
    if kpts.size < 17 * 3:
        return 0
    
    # 计算肩部中点（索引5:左肩, 6:右肩）
    left_shoulder = kpts[5][:2]
    right_shoulder = kpts[6][:2]
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    
    # 计算臀部中点（索引11:左臀, 12:右臀）
    left_hip = kpts[11][:2]
    right_hip = kpts[12][:2]
    hip_mid = (left_hip + right_hip) / 2
    
    # 计算身体中心线的角度
    dy = hip_mid[1] - shoulder_mid[1]
    dx = hip_mid[0] - shoulder_mid[0]
    
    if dx == 0:
        return 0
    
    angle = np.degrees(np.arctan(dy / dx))
    return abs(angle)

def main():
    # 加载模型
    print("加载模型中...")
    model = YOLO("GUI/ptfiles/model1.pt")  # 使用YOLOv8姿态模型
    
    # 设置参数
    tilt_threshold = 15  # 倾斜角度阈值
    warning_time_threshold = 3  # 倾斜超过3秒发出警告
    
    # 初始化状态
    is_tilting = False
    tilt_start_time = None
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("身体倾斜检测已启动")
    print(f"当身体倾斜超过{tilt_threshold}度且持续{warning_time_threshold}秒以上时，将发出警报")
    print("按 'q' 键退出")
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 运行推理
        results = model(frame)
        
        # 处理结果
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # 计算倾斜角度
                angle = calculate_tilt_angle(result.keypoints)
                
                # 检查倾斜并发出警告
                if angle > tilt_threshold:
                    if not is_tilting:
                        is_tilting = True
                        tilt_start_time = time.time()
                    else:
                        # 计算倾斜持续时间
                        tilt_duration = time.time() - tilt_start_time
                        if tilt_duration > warning_time_threshold:
                            # 发出警告声
                            winsound.Beep(800, 500)  # 800Hz, 500ms
                            print(f"警告: 身体倾斜超过{warning_time_threshold}秒!")
                else:
                    # 重置状态
                    is_tilting = False
                    tilt_start_time = None
                
                # 在图像上显示倾斜角度
                cv2.putText(frame, f'Tilt Angle: {angle:.1f}°', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 绘制关键点和骨架
                result.plot(frame=frame)
        
        # 显示帧
        cv2.imshow('身体倾斜检测', frame)
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

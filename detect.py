"""
使用时一定注意摄像头与竖直方向夹45度角，同时摄像头与地面距离0.65米，这很重要！
"""

from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.yaml").load("runs/detect/train3/weights/best.pt")  # 加载模型

# YOLO检测种类字典
yolo_classes = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair dryer',
    79: 'toothbrush'
}

confidence_threshold = 0.35  # 设置置信度阈值
def detect():
    cap = cv2.VideoCapture(0)
    paused = False  # 用于控制暂停状态
    while 1:
        if not paused:
            ret, frame = cap.read()
            if not ret:  # 检查是否成功读取帧
                print("Error: Failed to capture image.")
                break
            results = model(frame)
            # img = results[0].plot()
            for i, (x, y, w, h) in enumerate(results[0].boxes.xywh):
                cls_id = int(results[0].boxes.cls[i])  # 获取类别ID
                confidence = results[0].boxes.conf[i]  # 获取置信度
                label = yolo_classes.get(cls_id, "未知")  # 获取中文标签（默认未知）
                if confidence > confidence_threshold:
                    cv2.rectangle(frame,
                                (int(x - w / 2), int(y - h / 2)),
                                (int(x + w / 2), int(y + h / 2)),
                                (0, 255, 0), 2)  # 绘制矩形框（绿色，线宽为2）
                # 在图像上绘制标签（用中文或英文）
                    cv2.putText(frame, label, (int(x - w/2), int(y - h/2 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    p = y + h / 2  # 像素坐标
                    distance_real = 1.218e-11*p*p*p*p - 2.259e-8*p*p*p + 1.759e-5*p*p - 0.007766*p + 1.924   #拟合函数
                    print(f"{label}:{distance_real}:{p}")  # 输出格式“种类：真实世界距离：像素距离”
            cv2.imshow('Detections', frame)
        key = cv2.waitKey(1) & 0xFF # 等待按键输入
        if key == ord('a'):  # 按下'a'结束
            break
        elif key == ord('b'):  # 按下‘b’键暂停/恢复
            paused = not paused
    cap.release()
    cv2.destroyAllWindows()

detect()

# 现实采集数据
data = {
    "height": 0.65,
    "angle": 45,
    "p":"real",
    1: 1.9,
    20: 1.8,
    32: 1.7,
    46: 1.6,
    62: 1.5,
    79: 1.4,
    100: 1.3,
    122: 1.2,
    147: 1.1,
    173: 1.0,
    207: 0.9,
    241: 0.8,
    280: 0.7,
    337: 0.6,
    400: 0.5,
    421: 0.47,
    435: 0.45,
    457: 0.42,
    480: 0.4
}
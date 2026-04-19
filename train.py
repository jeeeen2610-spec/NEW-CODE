import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 原yolov8s
    yaml_yolov8s = 'ultralytics/cfg/models/v8/yolov8s.yaml'
    # 旧的 SE 注意力机制
    yaml_yolov8_SE = 'ultralytics/cfg/models/v8/det_self/yolov8s-attention-SE.yaml'

    # 🌟 新增：2023年最新的 EMA 注意力机制路径
    yaml_yolov8_EMA = 'ultralytics/cfg/models/v8/det_self/yolov8s-EMA.yaml'

    # 指向新的 EMA 变量！
    model_yaml = yaml_yolov8_EMA

    # 模型加载
    model = YOLO(model_yaml)

    # 数据集路径的yaml文件
    data_path = r'config\traindata.yaml'

    # 以yaml文件的名字进行命名 (跑完后，文件夹名字会自动变成 yolov8s-EMA)
    name = os.path.basename(model_yaml).split('.')[0]

    # 模型训练参数
    model.train(data=data_path,  # 数据集
                imgsz=640,  # 训练图片大小
                epochs=200,  # 训练的轮次
                batch=2,  # 训练batch (如果显存够，可以改成 4 或 8 跑得更快)
                workers=0,  # 加载数据线程数
                device='cpu',  # 使用cpu
                optimizer='SGD',  # 优化器
                project='runs/train',  # 模型保存路径
                name=name,  # 模型保存命名
                )

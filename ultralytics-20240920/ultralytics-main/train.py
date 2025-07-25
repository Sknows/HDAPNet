import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolo/ultralytics-20240920/ultralytics-main/ultralytics/cfg/models/v8/yolov8-bifpn_neck_dif.yaml') 
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/yolo/ultralytics-main/VisDrone/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=8,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                #resume=True, # 断点续训,YOLO初始化时选择last.pt
                 #amp=False, # close amp
                #half=False,
                # fraction=0.2,
                project='runs/train',
                name='yolov8_twomodel',
                )

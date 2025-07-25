# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolo/ultralytics-20240920/ultralytics-main/runs/train/HDAPNet/weights/best.pt')
    model.val(data='/root/autodl-tmp/yolo/ultralytics-main/VisDrone/data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              save=True,
              show_conf=False,
              show_labels=False,
              project='runs/val',
              name='best',
              )
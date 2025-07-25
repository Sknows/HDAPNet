import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolo/ultralytics-20240920/ultralytics-main/yolov8n.pt') # select your model.pt path
    model.predict(source='/root/autodl-tmp/yolo/ultralytics-main/VisDrone/VisDrone2019-DET-val/images',
                  imgsz=640,
                  project='runs/detect',
                  name='Yolov11noconf',
                  save=True,
                  conf=0.2,
                  iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  line_width=1, # line width of the bounding boxes
                  show_conf=False, # do not show prediction confidence
                  show_labels=False, # do not show prediction labels
                  save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )
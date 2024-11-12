import os
import argparse
import numpy as np
import cv2
from mean_average_precision import MetricBuilder

from detectors import UltralytDetector, RKNNDetector


label_list = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
              "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", 
              "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
              "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
              "toothbrush"]

def yolo_to_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox

    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return [x1, y1, x2, y2]


def imgs_lbls(dataset_path):
    images = []
    anns = []

    files = os.listdir(dataset_path + '/images/')
    images = files
    files = os.listdir(dataset_path + '/labels/')
    anns = files

    images = sorted(images)
    anns = sorted(anns)
    return images, anns


def preds_gt(detector, dataset_path):
    gt = []
    preds = []

    images, anns = imgs_lbls(dataset_path)
    for i in range(len(images)):
        image = cv2.imread(f"{dataset_path}/images/{images[i]}")
        with open(f"{dataset_path}labels/{anns[i]}", "r") as f:
            try:
                results = detector.run(image)
                # [xmin, ymin, xmax, ymax, class_id, confidence]
                preds += results
                j = 0 
                for line in f.readlines():
                    ann = [float(item) for item in line.split(" ")] # str2float
                    h, w, _ = image.shape
                    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
                    ann = [yolo_to_bbox(ann[1:], w, h) + [int(ann[0])] + [0, 0]]
                    gt += [ann]
            except:
                continue
    return preds, gt


def per_classes(data_list):
    classes_group = dict()
    for obj_data in data_list:
        cls_id = int(obj_data[0][4])
        if classes_group.get(cls_id, False):
            classes_group[cls_id].append(obj_data[0])
        else:
            classes_group[cls_id] = [obj_data[0]]
    return classes_group


def evaluate(preds, gt):
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    class_id = preds[0][4]
    for pred in preds:
        pred[4] = 0
    for gth in gt:
        gth[4] = 0
    preds = np.array(preds, dtype=np.float32)
    gt = np.array(gt, dtype=np.float32)
    metric_fn.add(preds, gt)
    # metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    # metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
    return class_id, metric_fn.value(iou_thresholds=np.arange(0.5, 0.95, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']#metric_fn.value(iou_thresholds=0.5)['mAP']


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Inference Script')
    parser.add_argument('--model_path', type=str, required=True, help='Model path (.pt, .rknn, .onnx file)')
    parser.add_argument('--dataset_path', type=str, default='data/', help='Dataset folder path')
    args = parser.parse_args()
    return args


def get_detector(model_path):
    if not os.path.exists(model_path):
        print("Cannot load model. File not exists")
        return None
    if model_path.endswith('rknn'):
        detector = RKNNDetector(model_path)
        return detector
    else:
        detector = UltralytDetector(model_path)
        return detector


def main(args):
    detector = get_detector(args.model_path)
    if detector:
        preds, gt = preds_gt(detector, args.dataset_path)
        preds_dict = per_classes(preds)
        gt_dict = per_classes(gt)
        eval_results = []
        for k in preds_dict:
            if gt_dict.get(k, False):
                eval_results.append(evaluate(preds_dict[k], gt_dict[k]))

        print('class    mAP50')
        for cls_id, mAP50 in eval_results:
            print(label_list[cls_id], mAP50)


if __name__ == '__main__':
    args = parse_args()
    main(args)

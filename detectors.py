import cv2
import numpy as np


class UltralytDetector(): # host
    CONF_TH = 0.25
    IOU_TH = 0.6

    def __init__(self, model_path="yolov8s.pt"): # "yolov8s.onnx"
        from ultralytics import YOLO
        self.model = YOLO(model_path)  # load an official model


    def run(self, image):
        results = self.model.predict(image,  conf=self.CONF_TH)
        result = results[0]
        pred = []
        for box in result.boxes:
            x_min_pred, y_min_pred, x_max_pred, y_max_pred = box.xyxy[0]
            x_min_pred, y_min_pred, x_max_pred, y_max_pred = x_min_pred.cpu().numpy().item(), y_min_pred.cpu().numpy().item(), x_max_pred.cpu().numpy().item(), y_max_pred.cpu().numpy().item()
            bbox = [int(x_min_pred), int(y_min_pred), int(x_max_pred), int(y_max_pred)]
            conf_pred = [box.conf.cpu().numpy().item()]
            cls_id = [int(box.cls.cpu().numpy().item())]
            det = [bbox + cls_id + conf_pred]
            pred.append(det)
        return pred
        


class RKNNDetector: # rockchip
    CONF_TH = 0.25
    IOU_TH = 0.6

    def create_rknn_session(self, model_path, core_mask):
        from rknnlite.api import RKNNLite
        rknn_lite = RKNNLite()
        ret = rknn_lite.load_rknn(model_path)
        if ret:
            raise OSError(f"{model_path}: Export rknn model failed!")
        ret = rknn_lite.init_runtime(async_mode=True, core_mask=core_mask)
        if ret:
            raise OSError(f"{model_path}: Init runtime enviroment failed!")
        return rknn_lite

    def __init__(self, model_path: str, core_mask = 0):
        self.session = self.create_rknn_session(model_path, core_mask)
        self.net_size = 640

    @staticmethod
    def letterbox(
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleup=True,
        stride=32,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:

        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, r, (dw, dh)

    def pre_process(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        img, ratio, dwdh = self.letterbox(
            img, new_shape=(self.net_size, self.net_size), auto=False
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img, ratio, dwdh

    def inference(self, img: np.ndarray) -> np.ndarray | None:
        return self.session.inference(inputs=[img])

    def filter_boxes(
        self,
        boxes: np.ndarray,
        box_confidences: np.ndarray,
        box_class_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter boxes with object threshold."""
        box_confidences = box_confidences.flatten()
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        scores = class_max_score * box_confidences
        mask = scores >= self.CONF_TH

        return boxes[mask], classes[mask], scores[mask]

    def dfl(self, position: np.ndarray) -> np.ndarray:
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        exp_y = np.exp(y)
        y = exp_y / np.sum(exp_y, axis=2, keepdims=True)
        acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
        return np.sum(y * acc_metrix, axis=2)

    def box_process(self, position: np.ndarray) -> np.ndarray:
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        grid = np.stack((col, row), axis=0).reshape(1, 2, grid_h, grid_w)
        stride = np.array([self.net_size // grid_h, self.net_size // grid_w]).reshape(
            1, 2, 1, 1
        )
        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy

    def post_process(
        self, outputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        def sp_flatten(_in):
            ch = _in.shape[1]
            return _in.transpose(0, 2, 3, 1).reshape(-1, ch)
        defualt_branch = 3
        pair_per_branch = len(outputs) // defualt_branch
        boxes, classes_conf, scores = [], [], []
        for i in range(defualt_branch):
            boxes.append(self.box_process(outputs[pair_per_branch * i]))
            classes_conf.append(sp_flatten(outputs[pair_per_branch * i + 1]))
            scores.append(np.ones_like(classes_conf[-1][:, :1], dtype=np.float32))
        boxes = np.concatenate([sp_flatten(b) for b in boxes])
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores).flatten()
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.CONF_TH, self.IOU_TH
        )
        if isinstance(indices, tuple):
            return None, None, None
        boxes = boxes[indices]
        classes = classes[indices]
        scores = scores[indices]
        return boxes, classes, scores

    def run(self, img: np.ndarray) -> list:
        dets = []
        pre_img, ratio, dwdh = self.pre_process(img)

        outputs = self.inference(pre_img)
        if outputs is not None:
            boxes, classes, scores = self.post_process(outputs)
            if boxes is not None:
                boxes -= np.array(dwdh * 2)
                boxes /= ratio
                boxes = boxes.round().astype(np.int32)
                for box, score, cl in zip(boxes, scores, classes):
                    x0, y0, x1, y1 = map(int, box)
                    #print('x0, y0, x1, y1, cl, score', x0, y0, x1, y1, cl, score)
                    dets.append([x0, y0, x1, y1, cl, score])
        return [dets]


if __name__ == '__main__':
    detector = UltralytDetector()
    img = cv2.imread("data/images/000000000009.jpg")
    pred = detector.run(img)
    print(pred)

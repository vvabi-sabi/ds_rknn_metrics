# RKNN metrics

This repository allows you to evaluate results using mAP metrics.

## Usage

### Running Inference

Use `main.py` to perform object detection inference on a set of images.

#### Command-Line Arguments

- `--model_path` (str, required): Path to the model file (`.rknn`, `.onnx`).
- `--dataset_path` (str): Path to validation-images and YOLO annotation files (labels) (default: `data/`).

#### Example Command

```bash
python main.py \
    --model_path models/yolov8.rknn \
    --dataset_path data/
```

## Example

1. **Prepare Data**

   - Place your images in `data/images`.
   - Place your annotations `.txt` files in `data/labels`.

2. **Configure detector**

   Adjust thresholds, input-image sizes, and class labels as needed.

3. **Run Inference and Evaluation**

   ```bash
   python main.py \
       --model_path models/yolov8.rknn \
       --dataset_path data/ 
   ```

4. **View Results**
   - mAP metrics will be printed in the console.

import streamlit as st
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from collections import defaultdict
import torch
import tempfile
# Load the ResNet model
model = models.resnet50(weights='imagenet')

model.eval()

# Define image transformation
imgsz = 640
transform = transforms.Compose([
    transforms.Resize((imgsz, imgsz)),
    transforms.ToTensor(),
])

# Function to parse ResNet output into a DataFrame
def parse_results(output):
    _, predicted_class = output.max(1)
    confidence = output.softmax(1).max().item()
    data = {'name': [predicted_class.item()], 'confidence': [confidence]}
    return pd.DataFrame(data)

# Function to load ground truth annotations
def load_ground_truth_annotations(image_file, labels_dir):
    label_file = os.path.join(labels_dir, image_file.replace(".jpg", ".txt"))
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        annotations = []
        for line in lines:
            class_index, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = max(0, (x_center - width / 2))
            y_min = max(0, (y_center - height / 2))
            x_max = min(1, (x_center + width / 2))
            y_max = min(1, (y_center + height / 2))
            annotation = {
                "class": int(class_index),
                "bbox": (x_min, y_min, x_max, y_max)
            }
            annotations.append(annotation)
        return annotations
    else:
        return []

# Function to compare prediction with ground truth
def compare_prediction(prediction, ground_truth, iou_threshold=0.5):
    class_precision = defaultdict(float)
    class_recall = defaultdict(float)
    class_true_negatives = defaultdict(int)

    for gt_obj in ground_truth:
        gt_class = gt_obj["class"]
        gt_bbox = gt_obj["bbox"]
        class_tp_found = False
        for _, p_obj in prediction.iterrows():
            p_class = p_obj['name']
            p_bbox = (p_obj['xmin'], p_obj['ymin'], p_obj['xmax'], p_obj['ymax'])
            iou = calculate_iou(gt_bbox, p_bbox)
            if iou >= iou_threshold and p_class == gt_class:
                class_tp_found = True
                break
        if class_tp_found:
            class_precision[gt_class] += 1
        else:
            class_recall[gt_class] += 1

    for gt_obj in ground_truth:
        gt_class = gt_obj["class"]
        class_true_negatives[gt_class] += len(prediction) - 1  

    total_precision = sum(class_precision.values()) / len(class_precision) if len(class_precision) > 0 else 0
    total_recall = sum(class_recall.values()) / len(ground_truth) if len(ground_truth) > 0 else 0
    total_true_negatives = sum(class_true_negatives.values())

    return {
        "precision": total_precision,
        "recall": total_recall,
        "true_negatives": total_true_negatives,
        "class_wise_metrics": {
            cls: {
                "precision": class_precision[cls],
                "recall": class_recall[cls],
                "true_negatives": class_true_negatives[cls]
            }
            for cls in class_precision
        }
    }

# Function to calculate IoU
def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou

# Streamlit app
def main():
    st.title("Object Detection Model Evaluation")

    # Upload folder of images
    uploaded_folder = st.file_uploader("Upload a folder of images",  type=["zip"])
    if uploaded_folder:
        st.write("Images uploaded successfully!")

        # Create a temporary directory to store uploaded files
        temp_dir = tempfile.TemporaryDirectory()
        uploaded_files = st.file_uploader("Upload a folder of images",  type=["zip"])

        if uploaded_files:
            # Save uploaded files to temporary directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir.name, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Process each image in the temporary directory
            for image_file in os.listdir(temp_dir.name):
                if image_file.endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(temp_dir.name, image_file)

                    # Load the image
                    image = Image.open(image_path).convert("RGB")

                    # Preprocess the image
                    input_tensor = transform(image).unsqueeze(0)

                    # Perform inference
                    with torch.no_grad():
                        output = model(input_tensor)

                    # Parse model output
                    prediction = parse_results(output)

                    # Load ground truth annotations
                    labels_dir = ""  # Define the directory containing ground truth annotations
                    ground_truth = load_ground_truth_annotations(image_file, labels_dir)

                    # Compare predictions with ground truth
                    metrics = compare_prediction(prediction, ground_truth)

                    # Display results
                    st.write(f"Image: {image_file}")
                    st.write(f"Metrics: {metrics}")

        # Cleanup: Close and remove temporary directory
        temp_dir.cleanup()
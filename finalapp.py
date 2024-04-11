import streamlit as st
from PIL import Image
import numpy as np
import os
import zipfile
import io

# Import your model and other necessary functions
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img): 
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    prediction["image"] = img  # Add the image to the prediction dictionary
    prediction["info"] = "Additional information"  # Add additional information
    return prediction

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 (list): Coordinates [x1, y1, x2, y2] of the first bounding box.
        box2 (list): Coordinates [x1, y1, x2, y2] of the second bounding box.

    Returns:
        float: Intersection over Union (IoU) value.
    """
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate Intersection over Union (IoU)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def calculate_metrics(prediction, ground_truth):
    true_positives = false_positives = true_negatives = false_negatives = 0

    for pred_box in prediction["boxes"]:
        if all(iou(pred_box, gt_box) < 0.5 for gt_box in ground_truth["boxes"]):
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    return metrics



# Function to process the images within the zip file
def process_zip_images(zip_file, ground_truth):
    predictions = []
    metrics = []

    with zipfile.ZipFile(zip_file) as z:
        for filename in z.namelist():
            if filename.endswith((".jpg", ".png", ".jpeg")):
                with z.open(filename) as f:
                    img = Image.open(io.BytesIO(f.read()))

                # Perform prediction using your model and other necessary functions
                prediction = make_prediction(img)
                metric = calculate_metrics(prediction, ground_truth)

                predictions.append(prediction)
                metrics.append(metric)

    return predictions, metrics

# Streamlit app
st.title("Object Detector :tea: :coffee:")
uploaded_zip = st.file_uploader("Upload Zip File Containing Images:", type=["zip"])
if uploaded_zip:
    # Define the ground truth data
    ground_truth = {
        "labels": [1, 0, 1, 0, 1],
        "boxes": [[10, 10, 50, 50], [60, 60, 100, 100], [20, 20, 70, 70], [80, 80, 120, 120], [30, 30, 90, 90]]
    }

    predictions, metrics = process_zip_images(uploaded_zip, ground_truth)

    # Display predictions and metrics for each image
    for idx, (prediction, metric) in enumerate(zip(predictions, metrics)):
        st.header(f"Image {idx+1} Predictions")
        st.write(prediction)

        st.header(f"Image {idx+1} Model Metrics")
        st.write(metric)

    # Display aggregated metrics
    if metrics:
        avg_precision = np.mean([metric["precision"] for metric in metrics])
        avg_recall = np.mean([metric["recall"] for metric in metrics])
        avg_f1_score = np.mean([metric["f1_score"] for metric in metrics])

        st.header("Aggregated Model Evaluation Metrics")
        st.write(f"Average Precision: {avg_precision}")
        st.write(f"Average Recall: {avg_recall}")
        st.write(f"Average F1 Score: {avg_f1_score}")

# Add the remaining part of your Streamlit app here
# (e.g., defining other UI elements, handling user inputs, etc.)



# Function to process the images within the zip file
def process_zip_images(zip_file):
    predictions = []
    metrics = []

    with zipfile.ZipFile(zip_file) as z:
        for filename in z.namelist():
            if filename.endswith((".jpg", ".png", ".jpeg")):
                with z.open(filename) as f:
                    img = Image.open(io.BytesIO(f.read()))

                # Perform prediction using your model and other necessary functions
                prediction = make_prediction(img)
                metric = calculate_metrics(prediction)

                predictions.append(prediction)
                metrics.append(metric)

    return predictions, metrics


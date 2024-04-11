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


def calculate_metrics(prediction):
    ground_truth = {
        "labels": [1, 0, 1, 0, 1],
        "boxes": [[10, 10, 50, 50], [60, 60, 100, 100], [20, 20, 70, 70], [80, 80, 120, 120], [30, 30, 90, 90]]
    }

    if "labels" in prediction and "boxes" in prediction:
        pred_labels = prediction["labels"]
        pred_boxes = prediction["boxes"]

        tp = fp = fn = 0
        for idx, pred_label in enumerate(pred_labels):
            if pred_label == 1:
                if pred_boxes[idx] in ground_truth["boxes"]:
                    tp += 1
                else:
                    fp += 1
            else:
                if pred_boxes[idx] not in ground_truth["boxes"]:
                    fn += 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    else:
        metrics = {
            "precision": 0,
            "recall": 0,
            "f1_score": 0
        }

    return metrics


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

# Streamlit app
st.title("Object Detector :tea: :coffee:")
uploaded_zip = st.file_uploader("Upload Zip File Containing Images:", type=["zip"])

if uploaded_zip:
    predictions, metrics = process_zip_images(uploaded_zip)

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

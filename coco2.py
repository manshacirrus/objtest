import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"] ## ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stopsign',]
img_preprocess = weights.transforms() ## Scales values from 0-255 range to 0-1 range.

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval(); ## Setting Model for Evaluation/Prediction   
    return model

model = load_model()

def make_prediction(img): 
    img_processed = img_preprocess(img) ## (3,500,500) 
    prediction = model(img_processed.unsqueeze(0)) # (1,3,500,500)
    prediction = prediction[0]                       ## Dictionary with keys "boxes", "labels", "scores".
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction): ## Adds Bounding Boxes around original Image.
    img_tensor = torch.tensor(img) ## Transpose
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="person" else "green" for label in prediction["labels"]] , width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### (3,W,H) -> (W,H,3), Channel first to channel last.
    return img_with_bboxes_np

def calculate_metrics(prediction):
    # Define ground truth annotations (for demonstration purposes)
    ground_truth = {
        "labels": [1, 0, 1, 0, 1],  # Example ground truth labels
        "boxes": [[10, 10, 50, 50], [60, 60, 100, 100], [20, 20, 70, 70], [80, 80, 120, 120], [30, 30, 90, 90]]  # Example ground truth boxes
    }

    # Extract predicted labels and boxes if available
    if "labels" in prediction and "boxes" in prediction:
        pred_labels = prediction["labels"]
        pred_boxes = prediction["boxes"]

        # Calculate True Positives, False Positives, and False Negatives
        tp = fp = fn = 0
        for idx, pred_label in enumerate(pred_labels):
            if pred_label == 1:  # Assuming '1' represents the positive class
                if pred_boxes[idx] in ground_truth["boxes"]:
                    tp += 1
                else:
                    fp += 1
            else:
                if pred_boxes[idx] not in ground_truth["boxes"]:
                    tn += 1
                else:
                    fn += 1

        # Calculate precision, recall, and F1 score
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
## Dashboard
st.title("Object Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)

    prediction = make_prediction(img) ## Dictionary
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction) ## (W,H,3) -> (3,W,H)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)
    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]
    st.header("Predicted Probabilities")
    st.write(prediction)
    
    metrics = calculate_metrics(prediction)
    st.header("Model Evaluation Metrics")
    st.write(metrics)
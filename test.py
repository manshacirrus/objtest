import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the trained ResNet model
#model_path = r"C:\Users\Asus\Downloads\objtest\resnet_model.h5"
#resnet_model = load_model(model_path)
# Save the model in the correct format
#resnet_model.save("C:/Users/Asus/Downloads/objtest/resnet_model.h5")

# Function for object detection using the ResNet model
def detect_objects(image):
    # Preprocess the image
    image_resized = image.resize((140, 140))  # Resize the image to match model input size
    image_array = np.array(image_resized)
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    predictions = resnet_model.predict(image_array)  # Get predictions from the model
    
    # Calculate model metrics (example: mean prediction value)
    mean_prediction = np.mean(predictions)
    
    return predictions, mean_prediction

# Streamlit app
def main():
    st.title('Object Detection App with ResNet')
    
    uploaded_folder = st.file_uploader("Upload a folder of images", type=["zip"], accept_multiple_files=False)
    
    if uploaded_folder is not None:
        folder_path = "temp_folder"
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, uploaded_folder.name), "wb") as f:
            f.write(uploaded_folder.getvalue())
        
        st.write("Processing images in the folder...")
        image_files = os.listdir(folder_path)
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            st.image(image, caption=f'Image: {image_file}', use_column_width=True)
            
            object_detection_results, mean_prediction = detect_objects(image)
            # Display object detection results and model metrics for each image
            st.write(f"Object Detection Results for {image_file}:")
            st.write(object_detection_results)
            st.write(f"Mean Prediction Value for {image_file}:", mean_prediction)

if __name__ == '__main__':
    main()
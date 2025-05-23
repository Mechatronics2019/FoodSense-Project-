import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Get all the class names
with open("class_names.txt", "r") as f: 
    class_names = [food_name.strip() for food_name in  f.readlines()]


### 1. Creating Model and Load saved weights ###
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(class_names), 
)

effnetb2.load_state_dict(
    torch.load(
        f="trained_model.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 2. Creating predict function ###
 
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time

### 3. Building Gradio app ###

title = "FoodSense 🍔🍕🍩"
description = "An EfficientNetB2-based feature extractor designed for classifying food images into 101 distinct categories with high accuracy."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time(s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
)

# Launch the Gradio app
demo.launch()

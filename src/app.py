import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from model import create_model
import glob as glob

@st.cache
def process_image(image):
     # get the image file name for saving output later on
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    
    return outputs


@st.cache
def annotate_image(image, outputs, detection_threshold):
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            cv2.putText(image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 50, 50), 
                        1, lineType=cv2.LINE_AA)
            
            
        return image,pred_classes

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=6).to(device)
model.load_state_dict(torch.load(
    '../outputs/model20.pth', map_location=device
))
model.eval()

# classes: 0 index is reserved for background
CLASSES = [
    'background','crack' ,'damage' ,'pothole' ,'pothole_water' ,'pothole_water_m'
]

st.title("Road Damage Detection with Faster RCNN")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, 0.5 , 0.05
)

if img_file_buffer is not None:
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    detections = process_image(image)
    image, labels = annotate_image(image, detections, confidence_threshold)
    st.image(image, caption=f"Processed image", use_column_width=True)
    # st.write(labels)
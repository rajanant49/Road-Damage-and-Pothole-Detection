# Road-Damage-and-Pothole-Detection

This repository contains code for Road Damage and Pothole Detection using Faster RCNN in Pytorch.

* Train a cusom object detection model using the pre-trained PyTorch Faster R-CNN model with a ResNet-50-FPN backbone.
* The dataset that is used is the (https://www.kaggle.com/trolololo888/potholes-and-road-damage-with-annotations)[Road Damage and Pothole Detection] dataset from Kaggle.
* The dataset consisted of 500 images and corresponding annotation files in PascalVOC format. A split of 80:20 was used for training and validation data .
* For data augmentations , Albumentation library was used to introduce RandomRotate , MotionBlur , Flips and MedianBlur . 
* Created a simple yet very effective pipeline to fine-tune the PyTorch Faster RCNN model.
* After the training completes, carried out inference using new images from the internet that the model has not seen during training or validation.
* The model achieved a VOC PASCAL mAP of 0.4066.
* Inferenced Images can be seen in the test_predictions folder.

* The model is also deployed through streamlit where one can upload image of roads and observed various deformities detected by the model.

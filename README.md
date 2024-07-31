# Traffic Sign Detection

## Problem Statement
Develop a model that can automatically recognize traffic signs along the road.

## Model
We use the YOLOv5s model for this task due to its efficiency and accuracy in object detection.

## Data Collection and Preparation
1. **Download the dataset**: The dataset can be downloaded from [this link](https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1002-TrafSignDetc.zip).
2. Use **Roboflow** for data annotation and preparation.
3. **The different classes present are:**
![__results___27_0](https://github.com/user-attachments/assets/af9e635e-fe36-453a-a46e-fcb7838bde01)


## Setup Instructions

### 1. Clone YOLOv5 Repository
```python
# Clone the YOLOv5 GitHub repository
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
```
### 2. Install Dependencies
```python
!pip install -qr requirements.txt  # install dependencies
```
### 3. Download the Dataset
Add your Roboflow API key below to download the default traffic-sign-detection dataset
```python
# Install the roboflow library
!pip install roboflow

# Import the roboflow library and download the dataset
from roboflow import Roboflow

# Replace 'YOUR_API_KEY' with your actual Roboflow API key
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("traffic-sign-detection")
dataset = project.version(1).download("yolov5")
```
### 4. Prepare the dataset
```
# Move to the YOLOv5 directory
%cd /content/yolov5/

# View the location of the dataset
dataset.location
# View YAML file given by Roboflow containing class names
%cat {dataset.location}/data.yaml
```
### 5. Model Configuration and Architecture
```
# Define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
```
```
# YOLO model configuration
%cat /content/yolov5/models/yolov5s.yaml
```
##### Download all the pre-trained weights
```
!wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```
### 6. Train the Model
Train the YOLOv5s model on the custom dataset for n number of epochs.
```
# Train the model
!python train.py --img 640 --batch 16 --epochs 50 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```
### 7. Evaluate the Model
Evaluate the performance of the trained YOLOv5 object detector.
```
# Plot the training results
from utils.plots import plot_results
from IPython.display import Image

plot_results()
Image(filename='runs/train/exp/results.png', width=1000)
```
**Performance metrics:**
![performace-metrics](https://github.com/user-attachments/assets/567b54c8-8bf2-4264-9de6-1cd0b1b9151c)


### 8. Visualize Training Data
```
# Visualize training data with bounding box labels
from IPython.display import Image, display
Image(filename='/content/yolov5/runs/train/exp/train_batch0.jpg')
```
### 9. Run Inference with Trained Weights
```
# Run inference
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source data/images
```
### 10. Display Detected Images
```
# Display detected images
import glob
from IPython.display import Image, display

for imageName in glob.glob('runs/detect/exp/*.jpg'):
    display(Image(filename=imageName))
```
### 11. Conclusion
The YOLOv5s model demonstrates strong performance in detecting and classifying traffic signs. With further training and fine-tuning, the model can be optimized for real-world applications.





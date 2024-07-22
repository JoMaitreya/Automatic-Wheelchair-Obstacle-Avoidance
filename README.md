# Automatic Wheelchair Obstacle Detection and Avoidance

## Overview

This project uses OpenCV and a Raspberry Pi 4 to control a wheelchair based on object detection. The system uses a camera to detect objects and make real-time decisions to move the wheelchair. The primary objective is to navigate the wheelchair around obstacles and perform actions based on the detected objects without the need for human intervention.

### Components

- **Raspberry Pi:** Controls the GPIO pins for motor operation.
- **Camera Module:** Captures real-time video for object detection.
- **L298N Motor Driver:** Controls the wheelchair motors.
- **OpenCV:** Used for object detection and image processing.
- **TensorFlow SSD MobileNet:** Pre-trained model for object detection.

## Files

- `coco.names`: Class names for the COCO dataset.
- `frozen_inference_graph.pb`: Pre-trained weights for the SSD MobileNet model.
- `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`: Configuration file for the SSD MobileNet model.
- `detectMove.py`: Main script for object detection and motor control.
- `1-create.py`: (Deleted) Original script (if relevant, update as needed).
- `load_mod.py`: (Deleted) Original script (if relevant, update as needed).

## Setup

### Dependencies

Ensure you have the following installed:

- Python 3.x
- OpenCV
- NumPy
- RPi.GPIO

You can install the necessary Python packages using pip:

```sh
pip install opencv-python numpy RPi.GPIO
```
## Hardware Setup
Connect the Camera Module to the Raspberry Pi.
Connect the L298N Motor Driver to the Raspberry Pi GPIO pins:
out1 to GPIO 17
out2 to GPIO 18
out3 to GPIO 27
out4 to GPIO 22
Connect the motors to the L298N Motor Driver.
### Configuration
Download the pre-trained model and configuration files to your project directory:

coco.names
frozen_inference_graph.pb
ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
Update the paths in detectMove.py to reflect the location of these files.

## Usage
Run the script:
```sh
python detectMove.py
```
Wait for 10 seconds for the initial setup.

The wheelchair will start moving forward and perform actions based on detected objects.

## Troubleshooting
Camera Issues: Ensure the camera is properly connected and enabled.
GPIO Errors: Check the GPIO pin connections and ensure the L298N Motor Driver is functioning.
Dependencies: Make sure all required packages are installed.

## Acknowledgments
OpenCV Documentation

TensorFlow Object Detection API

Raspberry Pi Documentation
